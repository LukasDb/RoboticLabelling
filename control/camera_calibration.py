import numpy as np
import streamlit as st
import cv2
from typing import List
from model.scene import Scene
from model.camera.camera import Camera
from lib.geometry import *
import cv2
from tqdm import tqdm
from cv2 import aruco
from scipy import optimize


class CameraCalibrator:
    def __init__(self, scene: Scene, selected_camera):
        # needs to be persistent (which is not the case here)
        super().__init__()
        self._scene = scene
        self.captured_images: List[np.ndarray] = []
        self.captured_robot_poses: List[np.ndarray] = []
        self.selected_camera: Camera = None
        self.aruco_dict = None
        self.charuco_board = None

        self.mock_i = 0

    def setup(self):
        # TODO reset lighting
        if not self._scene.background.is_setup:
            self._scene.background.setup_window()
        self.setup_charuco_board()

    def select_camera(self, camera_name):
        cams = self._scene.cameras
        self.selected_camera = [x for x in cams if x.name == camera_name][0]

    def capture_image(self):
        # TODO actually capture image from camera
        # TODO choose the correct robot

        # mock_cam = "realsense_121622061798"
        mock_cam = "realsense_f1120593"
        from pathlib import Path

        for img_path in Path(f"demo_data/images/{mock_cam}").glob("*.png"):
            index = int(img_path.stem)
            pose_path = Path(f"demo_data/poses/{mock_cam}/{index}.txt")
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pose = np.loadtxt(str(pose_path))
            self.mock_i += 1
            self.captured_images.append(img)
            self.captured_robot_poses.append(pose)

        return
        # mock_img = np.random.uniform(size=(480, 640, 3), high=255).astype(np.uint8)
        # cv2.putText(
        #     mock_img,
        #     f"Captured Image {len(self.captured_images)}",
        #     (100, 100),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1,
        #     (0, 0, 0),
        #     2,
        #     cv2.LINE_AA,
        # )

        # pose = self._scene.robots[0].pose

        self.captured_images.append(img)
        self.captured_robot_poses.append(pose)

    def calibrate(self):
        if self.aruco_dict is None:
            print("Please setup charuco board first")
            return

        allCorners = []
        allIds = []
        decimator = 0
        markers = []
        # SUB PIXEL CORNER DETECTION CRITERION
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

        gray = None
        for img in tqdm(self.captured_images, desc="Detecting markers"):
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
                gray, self.aruco_dict
            )

            if len(corners) > 0:
                # SUB PIXEL DETECTION
                cv2.aruco.drawDetectedMarkers(img, corners, ids)
                for corner in corners:
                    cv2.cornerSubPix(
                        gray,
                        corner,
                        winSize=(3, 3),
                        zeroZone=(-1, -1),
                        criteria=criteria,
                    )
                markers.append(corners)
                _, inter_corners, inter_ids = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, gray, self.charuco_board
                )
                if (
                    inter_corners is not None
                    and inter_ids is not None
                    and len(inter_corners) > 3
                    and decimator % 1 == 0
                ):
                    allCorners.append(inter_corners)
                    allIds.append(inter_ids)

                    # draw onto img

            decimator += 1

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
        if ret:
            for img, rvec, tvec in zip(self.captured_images, rvecs, tvecs):
                cv2.drawFrameAxes(
                    img,
                    camera_matrix,
                    dist_coeffs,
                    rvec,
                    tvec,
                    0.3,
                )
        print("Done")
        print("Intrensic Camera Matrix:\n", camera_matrix)

        print("Calibrating extrinsics...")
        camera_poses = [
            np.concatenate([tvec, rvec], axis=0)[:, 0]
            for tvec, rvec in zip(tvecs, rvecs)
        ]
        ret = self._optimize_handeye_matrix(camera_poses, self.captured_robot_poses)

        x = ret["x"][:6]
        extrinsic_matrix = invert_homogeneous(
            get_affine_matrix_from_6d_vector("xyz", x)
        )

        print("Done")
        print("Optimality", ret["optimality"])
        print("Cost: ", ret["cost"])
        print("Extrinsic Camera Matrix:\n", extrinsic_matrix)

        # TODO save calibration

    def get_live_img(self) -> np.ndarray:
        # return live image from camera with projected charuco board (if calibrated)
        # TODO get live img from camera
        # If calibrated: draw charuco board on image
        return np.random.uniform(size=(480, 640, 3), high=255).astype(np.uint8)

    def get_selected_img(self, index) -> np.ndarray | None:
        # return selected image from camera with projected charuco board from cv2 detection
        if index is None:
            return None
        # project cv2 charuco board detection and estimated pose on image
        return self.captured_images[index]

    def setup_charuco_board(self):
        width = self._scene.background.screen_width
        width_mm = self._scene.background.screen_width_mm
        height = self._scene.background.screen_height
        height_mm = self._scene.background.screen_height_mm

        # MOCK FOR NOW
        height = 2160
        width = 3840
        width_mm = 16 / np.sqrt(16**2 + 9**2) * 0.8001 * 1000
        height_mm = 9 / np.sqrt(16**2 + 9**2) * 0.8001 * 1000

        pixel_w = width_mm / width / 1000.0
        pixel_h = height_mm / height / 1000.0
        print(f"Pixel dimensions: {pixel_w} x {pixel_h} m")

        chessboardSize = 0.05 // pixel_w * pixel_w  # in m
        # pixel_size * 126  # old value: 0.023 # [m]
        markerSize = 0.04 // pixel_w * pixel_w
        n_markers = (7, 5)  # x,y

        # create an appropriately sized image
        charuco_img_width = n_markers[0] * chessboardSize  # in m
        charuco_img_width = charuco_img_width / width_mm * width * 1000  # in pixel

        charuco_img_height = n_markers[1] * chessboardSize  # in m
        charuco_img_height = charuco_img_height / height_mm * height * 1000

        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.charuco_board = aruco.CharucoBoard.create(
            n_markers[0], n_markers[1], chessboardSize, markerSize, self.aruco_dict
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

        print(
            f"Confirm the dimensions of the chessboard in the image: {chessboardSize}"
        )
        print(f"Confirm the dimensions of the markers in the image: {markerSize}")

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
