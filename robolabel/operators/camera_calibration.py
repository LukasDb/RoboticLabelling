import numpy as np
import cv2
from cv2 import aruco  # type: ignore
from scipy import optimize
from scipy.spatial.transform import Rotation as R
import logging
import dataclasses
import numpy.typing as npt
from typing import Callable, Any

import robolabel as rl
from robolabel import Event
from robolabel.lib.geometry import (
    invert_homogeneous,
    get_affine_matrix_from_6d_vector,
)


@dataclasses.dataclass
class CalibrationDatapoint:
    img: npt.NDArray[np.uint8]
    robot_pose: npt.NDArray[np.float64] | None
    detected: bool = False
    estimated_pose6d: npt.NDArray[np.float64] | None = None
    corners: list[tuple[int]] = dataclasses.field(default_factory=lambda: [])
    inter_corners: list[tuple[float]] = dataclasses.field(default_factory=lambda: [])
    ids: list[list[int]] = dataclasses.field(default_factory=lambda: [])
    inter_ids: list[list[int]] = dataclasses.field(default_factory=lambda: [])


class CameraCalibrator(rl.Observer):
    def __init__(self, scene: rl.Scene):
        # needs to be persistent (which is not the case here)
        super().__init__()
        self._scene = scene
        self.listen_to(self._scene)

        self.chessboard_size = 0.05
        self.marker_size = 0.04
        self.MARKERS = (7, 5)
        self.CHARUCO_DICT: Any = aruco.DICT_6X6_250

        self.is_active = False
        self.calibration_datapoints: list[CalibrationDatapoint] = []
        self.mock_i = 0
        self.camera: rl.camera.Camera | None = self._scene.selected_camera
        self.markers2monitor = np.eye(4)
        self.trajectory_generator = rl.operators.TrajectoryGenerator()
        self.acquisition = rl.operators.Acquisition()
        self.setup_charuco_board()

    def update_observer(
        self, subject: rl.Observable, event: Event, *args: list[Any], **kwargs: dict[str, Any]
    ):
        if event == Event.MODE_CHANGED:
            assert "mode" in kwargs
            assert isinstance(kwargs["mode"], str)
            if kwargs["mode"] == "calibration":
                self.setup_charuco_board()
                self.reset()
                self.is_active = True
            else:
                self.is_active = False

        if not self.is_active:
            return

        if event == Event.CAMERA_SELECTED:
            if len(self.calibration_datapoints) != 0:
                logging.error("Calibration resetted because of camera switch.")
                self.reset()
            assert "camera" in kwargs
            assert isinstance(kwargs["camera"], rl.camera.Camera)
            self.camera = kwargs["camera"]
            self.setup_charuco_board()

    def reset(self) -> None:
        self.calibration_datapoints.clear()

    async def load(self, cal_data: dict[str, Any]) -> None:
        """load calibration from config dict"""
        self._scene.background.pose = np.array(cal_data["background_pose"])

        for c in self._scene.cameras.values():
            try:
                intrinsic_matrix = np.array(cal_data[c.unique_id]["intrinsic"])
                dist_coefficients = np.array(cal_data[c.unique_id]["dist_coefficients"])
                extrinsic_matrix = np.array(cal_data[c.unique_id]["extrinsic"])
                robot = cal_data[c.unique_id]["attached_to"]
                if robot != "none":
                    await c.attach(self._scene.robots[robot])
                c.set_calibration(intrinsic_matrix, dist_coefficients, extrinsic_matrix)
            except KeyError:
                pass
                # logging.info("No calibration data for camera", c.unique_id)

    async def dump(self) -> dict[str, Any]:
        """dump calibration as config dict"""
        cal_data = {}
        for cam in self._scene.cameras.values():
            if not cam.is_calibrated():
                continue

            cal_data[cam.unique_id] = {
                "intrinsic": cam.intrinsic_matrix.tolist(),
                "dist_coefficients": cam.dist_coefficients.tolist(),
                "extrinsic": cam.extrinsic_matrix.tolist(),
                "attached_to": "none" if cam.robot is None else cam.robot.name,
            }

        cal_data.update({"background_pose": (await self._scene.background.pose).tolist()})
        return cal_data

    def setup(self) -> None:
        # TODO reset lighting
        self._scene.background.setup_window()
        self.setup_charuco_board()

    def set_initial_guess(self, x, y, z):
        self.initial_guess = np.array([x, y, z])
        logging.debug(f"Changed guess for background monitor to {self.initial_guess}")

    async def run_self_calibration(self):
        assert self.camera is not None, "No camera selected"
        assert hasattr(
            self.camera, "run_self_calibration"
        ), "Camera does not support self calibration"
        assert self.camera.robot is not None, "Camera is not attached to a robot"

        target_position = self.initial_guess.copy()
        target_position[2] += 0.6  # 60cm above monitor

        target_pose = np.eye(4)
        target_pose[:3, 3] = target_position

        target_pose[:3, :3] = R.from_euler("y", 180, degrees=True).as_matrix()

        if self.camera.is_calibrated():
            # we can improve the position by using the calibration monitor pose
            target_position = (await self._scene.background.get_position()).copy()
            target_position[2] += 0.6  # 60cm above monitor
            target_pose[:3, 3] = target_position

            # and adjust for the camera flange link matrix
            target_pose = target_pose @ invert_homogeneous(self.camera.extrinsic_matrix)

        self._scene.background.set_textured()
        if not await self.camera.robot.move_to(target_pose, timeout=50):
            logging.error("Robot could not reach target position")
            return

        self.camera.run_self_calibration()  # type: ignore

    async def capture(self):
        if self.camera is None:
            logging.error("No camera selected")
            return
        if self.camera.robot is None:
            logging.error("Camera is not attached to a robot")
            return

        # since the depth is not used anyway
        robot_pose = await self.camera.robot.pose
        frame = self.camera.get_frame(depth_quality=rl.camera.DepthQuality.FASTEST)

        assert frame.rgb is not None, "No image captured"
        img = frame.rgb

        if img is None:
            logging.error("No image captured")
            return

        if robot_pose is None:
            logging.error("No robot pose available")
            return

        calibration_result = self._detect_charuco(img, robot_pose)

        self.calibration_datapoints.append(calibration_result)

    async def capture_images(self, cb: Callable | None = None):
        assert self.camera is not None, "No camera selected"
        trajectory_settings = rl.operators.TrajectorySettings(
            n_steps=20, view_jitter=0.0, z_cutoff=0.5, r_range=(0.4, 0.6)
        )
        trajectory = self.trajectory_generator.generate_trajectory_above_center(
            self.initial_guess, trajectory_settings
        )

        # await self.trajectory_generator.visualize_trajectory(self.camera, [])

        logging.debug("Starting acquisition for calibration...")

        i = 0
        async for _ in self.acquisition.execute([self.camera], trajectory):
            i += 1
            logging.debug(f"Reached {i}/{len(trajectory)}")
            if cb is not None:
                cb()
            await self.capture()

        if cb is not None:
            cb()

    def calibrate(self):
        if self.aruco_dict is None:
            logging.error("Please setup charuco board first")
            return

        if self.camera is None:
            logging.error("No camera selected")
            return

        inter_corners = [x.inter_corners for x in self.calibration_datapoints]
        inter_ids = [x.inter_ids for x in self.calibration_datapoints]
        robot_poses = [
            x.robot_pose for x in self.calibration_datapoints if x.robot_pose is not None
        ]
        assert len(robot_poses) == len(self.calibration_datapoints), "Robot poses missing"

        assert len(inter_corners) > 0, "No charuco corners detected"

        image_size = (self.camera.height, self.camera.width)

        cameraMatrixInit = np.array(
            [
                [2500.0, 0.0, image_size[1] / 2.0],
                [0.0, 2500.0, image_size[0] / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )

        distCoefficientsInit = np.zeros((5, 1))
        flags = (
            cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO  # type: ignore
        )
        # flags = (cv2.CALIB_RATIONAL_MODEL)
        logging.info("Calibrating intrinsics...")
        (
            ret,
            camera_matrix,
            dist_coefficients,
            rvecs,
            tvecs,
            _,
            _,
            _,
        ) = cv2.aruco.calibrateCameraCharucoExtended(  # type: ignore
            charucoCorners=inter_corners,
            charucoIds=inter_ids,
            board=self.charuco_board,
            imageSize=image_size,
            cameraMatrix=cameraMatrixInit,
            distCoeffs=distCoefficientsInit,
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9),  # type: ignore
        )
        assert ret, "Calibration failed"

        for cal_result, rvec, tvec in zip(self.calibration_datapoints, rvecs, tvecs):
            cal_result.estimated_pose6d = np.concatenate([tvec, rvec], axis=0)[:, 0].astype(
                np.float64
            )

        logging.info("Done")

        logging.info("Calibrating extrinsics...")
        camera_poses = [
            np.concatenate([tvec, rvec], axis=0)[:, 0] for tvec, rvec in zip(tvecs, rvecs)
        ]
        ret = self._optimize_handeye_matrix(camera_poses, robot_poses)
        logging.info("Done")
        logging.info(f"Optimality: {ret['optimality']}")
        logging.info(f"Cost:       {ret['cost']}")

        x = ret["x"]
        extrinsic_matrix = invert_homogeneous(get_affine_matrix_from_6d_vector("xyz", x[:6]))
        world2markers = invert_homogeneous(get_affine_matrix_from_6d_vector("xyz", x[6:]))

        # try OpenCV version
        R_gripper2base = [x[:3, :3] for x in robot_poses]
        t_gripper2base = [x[:3, 3] for x in robot_poses]
        R_target2cam = rvecs
        t_target2cam = tvecs
        # R_cam2gripper = extrinsic_matrix[:3, :3]  # to be estimated
        # t_cam2gripper = extrinsic_matrix[:3, 3]  # to be estamated
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(  # type: ignore
            R_gripper2base,
            t_gripper2base,
            R_target2cam,
            t_target2cam,
            method=cv2.CALIB_HAND_EYE_TSAI,  # type: ignore
        )

        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = R_cam2gripper
        extrinsic_matrix[:3, 3] = np.reshape(t_cam2gripper, (3,))

        self._scene.background.pose = world2markers @ self.markers2monitor

        # set camera atttributes
        self.camera.set_calibration(camera_matrix, dist_coefficients, extrinsic_matrix)

    async def draw_on_preview(self, cam: rl.camera.Camera, rgb: np.ndarray) -> np.ndarray:
        if not self.is_active or cam.robot is None:
            return rgb

        robot_pose = await cam.robot.pose if cam.robot is not None else None

        calibration_result = self._detect_charuco(rgb, robot_pose)
        rgb = await self._draw_cal_result(cam, calibration_result)
        return rgb

    async def get_from_image_cache(self, index) -> np.ndarray | None:
        if index is None:
            return None
        if self.camera is None:
            return None
        cal_result: CalibrationDatapoint = self.calibration_datapoints[index]
        img = await self._draw_cal_result(self.camera, cal_result)
        return img

    async def _draw_cal_result(self, cam: rl.camera.Camera, cal_result: CalibrationDatapoint):
        img = cal_result.img.copy()  # dont change original image in calibration results!

        if cal_result.detected:
            corners = cal_result.corners
            ids = cal_result.ids

            if len(corners) > 0:
                cv2.aruco.drawDetectedMarkers(img, corners, ids)  # type: ignore

        # if calibrated then also draw the monitor
        world2cam = await cam.pose
        if cam.is_calibrated():
            monitor = self._scene.background

            # draw optimized calibration result
            cam2monitor = invert_homogeneous(world2cam) @ (await monitor.pose)
            monitor.visualize_monitor_in_camera_view(
                img, cam.intrinsic_matrix, cam.dist_coefficients, cam2monitor
            )

            # draw per-image calibration result
            if cal_result.estimated_pose6d is not None:
                pose6d = cal_result.estimated_pose6d
                mat = get_affine_matrix_from_6d_vector("Rodriguez", pose6d)
                cam2monitor = mat @ self.markers2monitor
                monitor.visualize_monitor_in_camera_view(
                    img,
                    cam.intrinsic_matrix,
                    cam.dist_coefficients,
                    cam2monitor,
                    color=(255, 0, 0),
                )
        return img

    def setup_charuco_board(self):
        width = self._scene.background.screen_width
        width_m = self._scene.background.screen_width_m
        height = self._scene.background.screen_height
        height_m = self._scene.background.screen_height_m
        chessboard_size = self.chessboard_size
        marker_size = self.marker_size

        if isinstance(self.camera, rl.camera.DemoCam):
            # overwrite to monitor from lab (old demo data)
            logging.warn("USING MOCK MONITOR DIMENSIONS")
            height = 2160
            width = 3840
            diagonal_16_by_9 = np.linalg.norm((16, 9))
            width_m = 16 / diagonal_16_by_9 * 0.800  # 0.69726
            height_m = 9 / diagonal_16_by_9 * 0.800  # 0.3922
            chessboard_size = 0.0286  # better 0.05 or bigger
            marker_size = 0.023

        pixel_w = width_m / width
        pixel_h = height_m / height
        logging.debug(f"Pixel dimensions: {pixel_w} x {pixel_h} m")

        chessboard_size_scaled = chessboard_size // pixel_w * pixel_w  # in m
        marker_size_scaled = marker_size // pixel_w * pixel_w
        n_markers = self.MARKERS  # x,y

        self.aruco_dict: Any = aruco.getPredefinedDictionary(self.CHARUCO_DICT)
        self.charuco_board: Any = aruco.CharucoBoard.create(
            n_markers[0], n_markers[1], chessboard_size, marker_size_scaled, self.aruco_dict
        )

        # create an appropriately sized image
        charuco_img_width_m = n_markers[0] * chessboard_size_scaled  # in m
        charuco_img_width = charuco_img_width_m / width_m * width  # in pixel

        charuco_img_height_m = n_markers[1] * chessboard_size_scaled  # in m
        charuco_img_height = charuco_img_height_m / height_m * height

        # charuco board is created with pixel_w as square size
        # the actual pixel dimensions can vary so image needs to stretched/compressed in y
        y_factor = pixel_h / pixel_w
        charuco_img_height *= y_factor

        logging.debug(f"Creating Charuco image with size: {charuco_img_width, charuco_img_height}")
        charuco_img = self.charuco_board.draw(
            (round(charuco_img_width), round(charuco_img_height))
        )

        hor_pad = round((width - charuco_img_width) / 2)
        vert_pad = round((height - charuco_img_height) / 2)
        charuco_img = cv2.copyMakeBorder(  # type: ignore
            charuco_img,
            vert_pad,
            vert_pad,
            hor_pad,
            hor_pad,
            cv2.BORDER_CONSTANT,  # type: ignore
            value=(0, 0, 0),
        )

        self._scene.background.set_image(charuco_img)

        # calculate transform to the center of the screen
        # same orientation, translated by half of charuco width and height
        self.markers2monitor = np.eye(4)
        self.markers2monitor[0, 3] = charuco_img_width_m / 2.0
        self.markers2monitor[1, 3] = charuco_img_height_m / 2.0

        logging.warn(
            f"Confirm the dimensions of the chessboard in the image: {chessboard_size_scaled}"
        )
        logging.warn(f"Confirm the dimensions of the markers in the image: {marker_size_scaled}")

    def _optimize_handeye_matrix(self, camera_poses, robot_poses):
        camera2tool_t = np.zeros((6,))
        camera2tool_t[5] = np.pi  # initialize with 180° around z
        marker2wc_t = np.zeros((6,))
        marker2camera_t = [
            invert_homogeneous(get_affine_matrix_from_6d_vector("Rodriguez", x))
            for x in camera_poses
        ]

        tool2wc_t = [invert_homogeneous(x) for x in robot_poses]  # already homogeneous matrix

        x0 = np.array([camera2tool_t, marker2wc_t]).reshape(12)

        def residual(x, tool2wc=None, marker2camera=None):
            camera2tool = get_affine_matrix_from_6d_vector("xyz", x[:6])
            marker2wc = get_affine_matrix_from_6d_vector("xyz", x[6:])
            return res_func(marker2camera, tool2wc, camera2tool, marker2wc)

        def res_func(marker2camera, tool2wc, camera2tool, marker2wc):
            res = []
            for i in range(len(marker2camera)):
                res += single_res_func(marker2camera[i], tool2wc[i], camera2tool, marker2wc)
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

    def _detect_charuco(
        self, img: np.ndarray, robot_pose: np.ndarray | None
    ) -> CalibrationDatapoint:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)  # type: ignore

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # type: ignore
        assert gray is not None, "No image captured"
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict)  # type: ignore

        cal_result = CalibrationDatapoint(img=img, robot_pose=robot_pose)

        if len(corners) > 0:
            # SUB PIXEL DETECTION
            # cv2.aruco.drawDetectedMarkers(img, corners, ids)
            for corner in corners:
                cv2.cornerSubPix(  # type: ignore
                    gray,
                    corner,
                    winSize=(3, 3),
                    zeroZone=(-1, -1),
                    criteria=criteria,
                )

            _, inter_corners, inter_ids = cv2.aruco.interpolateCornersCharuco(  # type: ignore
                corners, ids, gray, self.charuco_board
            )
            if inter_corners is not None and inter_ids is not None and len(inter_corners) > 3:
                cal_result.detected = True
                cal_result.corners = corners
                cal_result.inter_corners = inter_corners
                cal_result.ids = ids
                cal_result.inter_ids = inter_ids

        return cal_result
