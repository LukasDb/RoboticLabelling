import pickle
import numpy as np
from pathlib import Path
import streamlit as st
import cv2
from typing import List, Dict
from model.scene import Scene
from model.camera.camera import Camera
from lib.geometry import *
import cv2
from tqdm import tqdm
from cv2 import aruco
from scipy import optimize


class CameraCalibrator:
    CHESSBOARD_SIZE = 0.0286  # better 0.05 or bigger
    MARKER_SIZE = 0.023
    MARKERS = (7, 5)
    CHARUCO_DICT = aruco.DICT_6X6_250

    def __init__(self, scene: Scene):
        # needs to be persistent (which is not the case here)
        super().__init__()
        self._scene = scene
        self.captured_images: List[np.ndarray] = []
        self.captured_robot_poses: List[np.ndarray] = []

        self.calibration_results: List[Dict] = []

        self.aruco_dict = None
        self.charuco_board = None
        self.markers2monitor = np.eye(4)

        self.mock_i = 0

    def load(self, cal_data: Dict):
        """load calibration from config dict"""
        self._scene.background.pose = cal_data["background_pose"]

        for c in self._scene.cameras.values():
            try:
                intrinsic_matrix = cal_data[c.unique_id]["intrinsic"]
                dist_coeffs = cal_data[c.unique_id]["dist_coeffs"]
                extrinsic_matrix = cal_data[c.unique_id]["extrinsic"]
                parent = cal_data[c.unique_id]["attached_to"]
                if parent != "none":
                    c.attach(self._scene.robots[parent], extrinsic_matrix)
                c.set_calibration(intrinsic_matrix, dist_coeffs, extrinsic_matrix)
            except KeyError:
                print("No calibration data for camera", c.unique_id)

    def dump(self) -> Dict:
        """dump calibration as config dict"""
        cal_data = {
            c.unique_id: {
                "intrinsic": c.intrinsic_matrix,
                "dist_coeffs": c.dist_coeffs,
                "extrinsic": c._link_matrix,
                "attached_to": "none" if c.parent is None else c.parent.name,
            }
            for c in self._scene.cameras.values()
        }

        cal_data.update({"background_pose": self._scene.background.pose})
        return cal_data

    def setup(self):
        # TODO reset lighting
        self._scene.background.setup_window()
        self.setup_charuco_board()

    def capture_image(self):
        img = self._scene.selected_camera.get_frame().rgb
        pose = self._scene.selected_camera.parent.pose
        if img is not None and pose is not None:
            self.captured_images.append(img)
            self.captured_robot_poses.append(pose)
            self.calibration_results.append({"detected": False})

    def calibrate(self):
        if self.aruco_dict is None:
            print("Please setup charuco board first")
            return

        allCorners = []
        allIds = []
        allImgs = []
        allRobotPoses = []
        # SUB PIXEL CORNER DETECTION CRITERION
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

        gray = None
        for img, pose, cal_result in tqdm(
            zip(
                self.captured_images,
                self.captured_robot_poses,
                self.calibration_results,
            ),
            desc="Detecting markers",
        ):
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
                gray, self.aruco_dict
            )

            cal_result.update(
                {"detected": len(corners) > 0, "corners": corners, "ids": ids}
            )

            if len(corners) > 0:
                # SUB PIXEL DETECTION
                # cv2.aruco.drawDetectedMarkers(img, corners, ids)
                for corner in corners:
                    cv2.cornerSubPix(
                        gray,
                        corner,
                        winSize=(3, 3),
                        zeroZone=(-1, -1),
                        criteria=criteria,
                    )

                _, inter_corners, inter_ids = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, gray, self.charuco_board
                )
                if (
                    inter_corners is not None
                    and inter_ids is not None
                    and len(inter_corners) > 3
                ):
                    allCorners.append(inter_corners)
                    allIds.append(inter_ids)
                    allImgs.append(img)
                    allRobotPoses.append(pose)
            else:
                allCorners.append([])
                allIds.append([])
                allImgs.append(img)
                allRobotPoses.append(pose)

        assert len(allCorners) > 0, "No charuco corners detected"

        imsize = gray.shape

        cameraMatrixInit = np.array(
            [
                [2500.0, 0.0, imsize[1] / 2.0],
                [0.0, 2500.0, imsize[0] / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )

        distCoeffsInit = np.zeros((5, 1))
        flags = (
            cv2.CALIB_USE_INTRINSIC_GUESS
            + cv2.CALIB_RATIONAL_MODEL
            + cv2.CALIB_FIX_ASPECT_RATIO
        )
        # flags = (cv2.CALIB_RATIONAL_MODEL)
        print("Calibrating intrinsics...")
        (
            ret,
            camera_matrix,
            dist_coeffs,
            rvecs,
            tvecs,
            _,
            _,
            _,
        ) = cv2.aruco.calibrateCameraCharucoExtended(
            charucoCorners=allCorners,
            charucoIds=allIds,
            board=self.charuco_board,
            imageSize=imsize,
            cameraMatrix=cameraMatrixInit,
            distCoeffs=distCoeffsInit,
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9),
        )
        assert ret, "Calibration failed"

        for cal_result, rvec, tvec in zip(self.calibration_results, rvecs, tvecs):
            cal_result["estimated_pose6d"] = np.concatenate([tvec, rvec], axis=0)[:, 0]

        print("Done")

        print("Calibrating extrinsics...")
        camera_poses = [
            np.concatenate([tvec, rvec], axis=0)[:, 0]
            for tvec, rvec in zip(tvecs, rvecs)
        ]
        ret = self._optimize_handeye_matrix(camera_poses, allRobotPoses)
        print("Done")
        print("Optimality: ", ret["optimality"])
        print("Cost:       ", ret["cost"])

        x = ret["x"]
        extrinsic_matrix = invert_homogeneous(
            get_affine_matrix_from_6d_vector("xyz", x[:6])
        )
        world2markers = invert_homogeneous(
            get_affine_matrix_from_6d_vector("xyz", x[6:])
        )

        self._scene.background.pose = world2markers @ self.markers2monitor

        # set camera atttributes
        self._scene.selected_camera.set_calibration(
            camera_matrix, dist_coeffs, extrinsic_matrix
        )

    def draw_calibration(self, rgb) -> np.ndarray:
        monitor = self._scene.background
        cam = self._scene.selected_camera
        if cam is None:
            return rgb
        
        if cam.intrinsic_matrix is not None:
            cam2monitor = invert_homogeneous(cam.pose) @ monitor.pose
            monitor.draw_on_rgb(
                rgb,
                cam.intrinsic_matrix,
                cam.dist_coeffs,
                cam2monitor,
            )

        return rgb

    def get_selected_img(self, index) -> np.ndarray | None:
        if index is None:
            return None

        monitor = self._scene.background
        cam = self._scene.selected_camera

        img = self.captured_images[index].copy()

        if cam.intrinsic_matrix is None:
            return img

        # draw optimized calibration result
        world2robot = self.captured_robot_poses[index]
        world2cam = world2robot @ cam.extrinsic_matrix
        cam2monitor = invert_homogeneous(world2cam) @ monitor.pose
        monitor.draw_on_rgb(img, cam.intrinsic_matrix, cam.dist_coeffs, cam2monitor)

        cal_result = self.calibration_results[index]
        if not cal_result["detected"]:
            return img

        corners = cal_result["corners"]
        ids = cal_result["ids"]
        pose6d = cal_result["estimated_pose6d"]

        if len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(img, corners, ids)

        # draw per-image calibration result
        mat = get_affine_matrix_from_6d_vector("Rodriguez", pose6d)
        cam2monitor = mat @ self.markers2monitor
        monitor.draw_on_rgb(
            img,
            cam.intrinsic_matrix,
            cam.dist_coeffs,
            cam2monitor,
            color=(255, 0, 0),
        )

        return img

    def setup_charuco_board(self):
        width = self._scene.background.screen_width
        width_m = self._scene.background.screen_width_m
        height = self._scene.background.screen_height
        height_m = self._scene.background.screen_height_m

        pixel_w = width_m / width
        pixel_h = height_m / height
        # print(f"Pixel dimensions: {pixel_w} x {pixel_h} m")

        chessboard_size = self.CHESSBOARD_SIZE // pixel_w * pixel_w  # in m
        marker_size = self.MARKER_SIZE // pixel_w * pixel_w
        n_markers = self.MARKERS  # x,y

        # create an appropriately sized image
        charuco_img_width_m = n_markers[0] * chessboard_size  # in m
        charuco_img_width = charuco_img_width_m / width_m * width  # in pixel

        charuco_img_height_m = n_markers[1] * chessboard_size  # in m
        charuco_img_height = charuco_img_height_m / height_m * height

        # charuco board is created with pixel_w as square size
        # the actual pixel dimensions can vary so image needs to stretched/compressed in y
        y_factor = pixel_h / pixel_w
        print("Apply y factor: ", y_factor)
        charuco_img_height *= y_factor

        self.aruco_dict = aruco.getPredefinedDictionary(self.CHARUCO_DICT)
        self.charuco_board = aruco.CharucoBoard.create(
            n_markers[0], n_markers[1], chessboard_size, marker_size, self.aruco_dict
        )

        print(
            "Creating Charuco image with size: ", charuco_img_width, charuco_img_height
        )
        charuco_img = self.charuco_board.draw(
            (round(charuco_img_width), round(charuco_img_height))
        )

        hor_pad = round((width - charuco_img_width) / 2)
        vert_pad = round((height - charuco_img_height) / 2)
        charuco_img = cv2.copyMakeBorder(
            charuco_img,
            vert_pad,
            vert_pad,
            hor_pad,
            hor_pad,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

        self._scene.background.set_image(charuco_img)

        # calculate transform to the center of the screen
        # same orientation, translated by half of charco width and height
        self.markers2monitor = np.eye(4)
        self.markers2monitor[0, 3] = charuco_img_width_m / 2.0
        self.markers2monitor[1, 3] = charuco_img_height_m / 2.0

        print(
            f"Confirm the dimensions of the chessboard in the image: {chessboard_size}"
        )
        print(f"Confirm the dimensions of the markers in the image: {marker_size}")

    def _optimize_handeye_matrix(self, camera_poses, robot_poses):
        camera2tool_t = np.zeros((6,))
        camera2tool_t[5] = np.pi  # initialize with 180° around z
        marker2wc_t = np.zeros((6,))
        marker2camera_t = [
            invert_homogeneous(get_affine_matrix_from_6d_vector("Rodriguez", x))
            for x in camera_poses
        ]

        tool2wc_t = [
            invert_homogeneous(x) for x in robot_poses  # already homogeneous matrix
        ]

        x0 = np.array([camera2tool_t, marker2wc_t]).reshape(12)

        def residual(x, tool2wc=None, marker2camera=None):
            camera2tool = get_affine_matrix_from_6d_vector("xyz", x[:6])
            marker2wc = get_affine_matrix_from_6d_vector("xyz", x[6:])
            return res_func(marker2camera, tool2wc, camera2tool, marker2wc)

        def res_func(marker2camera, tool2wc, camera2tool, marker2wc):
            res = []
            for i in range(len(marker2camera)):
                res += single_res_func(
                    marker2camera[i], tool2wc[i], camera2tool, marker2wc
                )
            return np.array(res).reshape(16 * len(marker2camera))

        def single_res_func(marker2camera, tool2wc, camera2tool, marker2wc):
            res_array = marker2camera @ camera2tool @ tool2wc - marker2wc
            return [res_array.reshape((16,))]

        ret = optimize.least_squares(
            residual,
            x0,
            kwargs={"marker2camera": marker2camera_t, "tool2wc": tool2wc_t},
        )
        return ret
