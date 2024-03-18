from __future__ import annotations

import contextlib
import time
from pathlib import Path
from threading import Event, Thread
from typing import List, Tuple

import kinpy as kp
import numpy as np
import numpy as onp
import numpy.typing as npt
import pyrealsense2 as rs  # type: ignore
import tyro
import viser
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from scipy.spatial.transform import Rotation
from viser.extras import ViserUrdf

from robo_cam_calib.utils.kinova_utils import DeviceConnection


def transform_to_matrix(transform):
    """Converts a transform object to a 4x4 transformation matrix."""
    from scipy.spatial.transform import Rotation
    res = np.eye(4)
    quat_wxyz = transform.rot
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    rotation = Rotation.from_quat(np.array(quat_xyzw)).as_matrix()
    position = np.array(transform.pos)
    res[:3, :3] = rotation
    res[:3, 3] = position
    return res

class RobotController:
    def __init__(self):
        self.connection = None
        self.base_cyclic = None
        self.urdf_file = "assets/kortex_description/robots/gen3_robotiq_2f_85.urdf"
        self.chain = kp.build_chain_from_urdf(open(self.urdf_file).read())


    def connect_to_robot(self):
        device_connection = DeviceConnection(DeviceConnection.IP, port=DeviceConnection.TCP_PORT, credentials=(DeviceConnection.USERNAME, DeviceConnection.PASSWORD))
        router = device_connection.connect()
        self.base_cyclic = BaseCyclicClient(router)

    def get_newest_joint_angles(self, pad_gripper: bool = False, radians: bool = True) -> List[float]:
        if not self.base_cyclic:
            raise Exception("Not connected to the robot")
        feedback = self.base_cyclic.RefreshFeedback()
        joint_angles_deg = [actuator.position for actuator in feedback.actuators] + ([0.0] if pad_gripper else [])

        if radians:
            joint_angles_rad = [angle * onp.pi / 180 for angle in joint_angles_deg]
            return joint_angles_rad

        return joint_angles_deg

    def disconnect_from_robot(self):
        if self.connection:
            self.connection.disconnect()
            self.connection = None

    def get_camera_pose(self, ret_as_transf: bool = False) -> np.ndarray:
        # Get the camera pose using the robot's FK
        newest_joint_angles = self.get_newest_joint_angles(pad_gripper=False, radians=True)
        # pad to thirteen joints from current 7
        newest_joint_angles += [0.0] * 6
        out = self.chain.forward_kinematics(th=newest_joint_angles)['camera_depth_frame']
        if ret_as_transf:
            return out
        # convert from Transform to 4x4 matrix
        return transform_to_matrix(out)


def adjust_camera_orientation(pose_matrix):
    """
    Flip camera orientation while maintaining position.
    So the desired new x-axis is direction of negative of old z-axis,
    the desired new y-axis is direction of negative old y-axis
    the desired new z-axis is direction of negative old x-axis
    """

    t_opengl = pose_matrix[:3, 3]

    x_old = pose_matrix[:3, 0]
    y_old = pose_matrix[:3, 1]
    z_old = pose_matrix[:3, 2]

    # Compute the new axes directions
    x_new = -z_old  # New x-axis is the negative of the old z-axis
    y_new = -y_old  # New y-axis is the negative of the old y-axis
    z_new = -x_old  # New z-axis is the negative of the old x-axis

    # Construct the new rotation matrix
    R_opencv = np.column_stack((x_new, y_new, z_new))

    # Rebuild the pose matrix with adjusted orientation and original position
    pose_matrix_opencv = np.eye(4)
    pose_matrix_opencv[:3, :3] = R_opencv
    pose_matrix_opencv[:3, 3] = t_opengl
    return pose_matrix_opencv


def realSenseThread(stop_event: Event, viser_server, robot_controller: RobotController):
    with realsense_pipeline() as pipeline:
        while not stop_event.is_set():
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            positions, colors = point_cloud_arrays_from_frames(depth_frame, color_frame)

            # apply camera_pose transformation to positions
            camera_pose = robot_controller.get_camera_pose()
            camera_pose = adjust_camera_orientation(camera_pose)

            # plot camera pose in viser
            camera_pos = camera_pose[:3, 3]
            camera_rot = camera_pose[:3, :3]
            camera_xyzw = Rotation.from_matrix(camera_rot).as_quat()
            camera_wxyz = np.array([camera_xyzw[-1], camera_xyzw[0], camera_xyzw[1], camera_xyzw[2]])

            # add camera pose frame
            viser_server.add_frame(
                "/camera",
                wxyz=camera_wxyz,
                position=camera_pos,
                axes_length=0.1,
                axes_radius=0.0025,
            )

            # first convert to opencv coord
            positions[:, 1] *= 1
            positions[:, 2] *= -1

            positions = camera_pose @ np.concatenate([positions, np.ones((positions.shape[0], 1))], axis=1).T
            positions = positions[:3, :].T

            # Visualize point cloud with viser server
            viser_server.add_point_cloud("/realsense", points=positions, colors=colors, point_size=0.001)

            # Example sleep, adjust as necessary. You might not need this if wait_for_frames is blocking and paced by frame arrival.
            time.sleep(0.1)

@contextlib.contextmanager
def realsense_pipeline(fps: int = 30):
    """Context manager that yields a RealSense pipeline."""

    # Configure depth and color streams.
    pipeline = rs.pipeline()  # type: ignore
    config = rs.config()  # type: ignore

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)  # type: ignore
    config.resolve(pipeline_wrapper)

    config.enable_stream(rs.stream.depth, rs.format.z16, fps)  # type: ignore
    config.enable_stream(rs.stream.color, rs.format.rgb8, fps)  # type: ignore

    # Start streaming.
    pipeline.start(config)

    yield pipeline

    # Close pipeline when done.
    pipeline.stop()

def point_cloud_arrays_from_frames(
    depth_frame, color_frame
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
    """Maps realsense frames to two arrays.

    Returns:
    - A point position array: (N, 3) float32.
    - A point color array: (N, 3) uint8.
    """
    # Processing blocks. Could be tuned.
    point_cloud = rs.pointcloud()  # type: ignore
    decimate = rs.decimation_filter()  # type: ignore
    decimate.set_option(rs.option.filter_magnitude, 3)  # type: ignore

    # Downsample depth frame.
    depth_frame = decimate.process(depth_frame)

    # Map texture and calculate points from frames. Uses frame intrinsics.
    point_cloud.map_to(color_frame)
    points = point_cloud.calculate(depth_frame)

    # Get color coordinates.
    texture_uv = (
        np.asanyarray(points.get_texture_coordinates())
        .view(np.float32)
        .reshape((-1, 2))
    )
    color_image = np.asanyarray(color_frame.get_data())
    color_h, color_w, _ = color_image.shape

    # Note: for points that aren't in the view of our RGB camera, we currently clamp to
    # the closes available RGB pixel. We could also just remove these points.
    texture_uv = texture_uv.clip(0.0, 1.0)

    # Get positions and colors.
    positions = np.asanyarray(points.get_vertices()).view(np.float32)
    positions = positions.reshape((-1, 3))
    colors = color_image[
        (texture_uv[:, 1] * (color_h - 1.0)).astype(np.int32),
        (texture_uv[:, 0] * (color_w - 1.0)).astype(np.int32),
        :,
    ]
    N = positions.shape[0]

    assert positions.shape == (N, 3)
    assert positions.dtype == np.float32
    assert colors.shape == (N, 3)
    assert colors.dtype == np.uint8

    return positions, colors


def update_joint_angles(urdf: ViserUrdf, gui_joints: List[viser.GuiInputHandle[float]], robot_controller):
    newest_angles = robot_controller.get_newest_joint_angles(pad_gripper=True, radians=True)
    print(f"newest_angles: {newest_angles}")
    for gui, new_angle in zip(gui_joints, newest_angles):
        gui.value = new_angle
    urdf.update_cfg(onp.array(newest_angles))

def robot_update(urdf: ViserUrdf, gui_joints: List[viser.GuiInputHandle[float]], frequency: float, robot_controller):
    interval = 1.0 / frequency
    while True:
        update_joint_angles(urdf, gui_joints, robot_controller)
        time.sleep(interval)

def initialize_robot(viser_server, urdf_path: Path):
    urdf = ViserUrdf(viser_server, urdf_path)

    # Create joint angle sliders.
    gui_joints = []
    initial_angles = []
    for joint_name, (lower, upper) in urdf.get_actuated_joint_limits().items():
        lower = lower if lower is not None else -np.pi
        upper = upper if upper is not None else np.pi

        initial_angle = 0.0 if lower < 0 and upper > 0 else (lower + upper) / 2.0
        slider = viser_server.add_gui_slider(
            label=joint_name,
            min=lower,
            max=upper,
            step=1e-3,
            initial_value=initial_angle,
        )
        slider.on_update(
            lambda _: urdf.update_cfg(np.array([gui.value for gui in gui_joints]))
        )

        gui_joints.append(slider)
        initial_angles.append(initial_angle)

    # Create joint reset button.
    reset_button = viser_server.add_gui_button("Reset")

    @reset_button.on_click
    def _(_):
        for g, initial_angle in zip(gui_joints, initial_angles):
            g.value = initial_angle

    # Apply initial joint angles.
    urdf.update_cfg(np.array([gui.value for gui in gui_joints]))

    return gui_joints, urdf


# Use in main or another appropriate function
def main(urdf_path: Path) -> None:
    viser_server = viser.ViserServer()
    robot_controller = RobotController()
    robot_controller.connect_to_robot()

    stop_event = Event()
    rs_thread = Thread(target=realSenseThread, args=(stop_event, viser_server, robot_controller))
    rs_thread.start()

    gui_joints, urdf = initialize_robot(viser_server, urdf_path)

    # Start continuous update thread at 60Hz
    robot_updater_thread = Thread(target=robot_update, args=(urdf, gui_joints, 60.0, robot_controller), daemon=True)
    robot_updater_thread.start()

    try:
        while True:
            time.sleep(1)  # Main thread can do other work or simply wait
    except KeyboardInterrupt:
        stop_event.set()  # Signal the RealSense thread to stop
        rs_thread.join()  # Wait for the RealSense thread to finish
        robot_updater_thread.join()

    robot_controller.disconnect_from_robot()
    print("Done.")

if __name__ == "__main__":
    tyro.cli(main)