from __future__ import annotations

import contextlib
import time
from pathlib import Path
from threading import Event, Thread
from typing import List, Tuple

import numpy as np
import numpy as onp
import numpy.typing as npt
import pyrealsense2 as rs  # type: ignore
import tyro
import viser
from viser.extras import ViserUrdf

# Other necessary imports


def realSenseThread(stop_event: Event, viser_server):
    with realsense_pipeline() as pipeline:
        while not stop_event.is_set():
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            positions, colors = point_cloud_arrays_from_frames(depth_frame, color_frame)

            # Transformation and visualization logic here
            # Apply any necessary transformations to positions
            R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float32)
            positions = positions @ R.T

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

def get_newest_joint_angles() -> List[float]:
    # Implement logic to fetch or calculate the newest joint angles.
    # This is a placeholder function.
    # return onp.random.uniform(-onp.pi, onp.pi, size=8).tolist()
    return onp.zeros(8).tolist()

def update_joint_angles(urdf: ViserUrdf, gui_joints: List[viser.GuiInputHandle[float]]):
    newest_angles = get_newest_joint_angles()
    for gui, new_angle in zip(gui_joints, newest_angles):
        gui.value = new_angle
    urdf.update_cfg(onp.array(newest_angles))

def robot_update(urdf: ViserUrdf, gui_joints: List[viser.GuiInputHandle[float]], frequency: float):
    interval = 1.0 / frequency
    while True:
        update_joint_angles(urdf, gui_joints)
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

    stop_event = Event()
    rs_thread = Thread(target=realSenseThread, args=(stop_event, viser_server))
    rs_thread.start()

    gui_joints, urdf = initialize_robot(viser_server, urdf_path)

    # Start continuous update thread at 60Hz
    robot_updater_thread = Thread(target=robot_update, args=(urdf, gui_joints, 60.0), daemon=True)
    robot_updater_thread.start()

    try:
        while True:
            time.sleep(1)  # Main thread can do other work or simply wait
    except KeyboardInterrupt:
        stop_event.set()  # Signal the RealSense thread to stop
        rs_thread.join()  # Wait for the RealSense thread to finish
        robot_updater_thread.join()

    print("Done.")

if __name__ == "__main__":
    tyro.cli(main)