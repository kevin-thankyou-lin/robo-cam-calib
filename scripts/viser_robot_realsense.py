"""URDF visualizer

Requires yourdfpy and URDF. Any URDF supported by yourdfpy should work.

Examples:
- https://github.com/OrebroUniversity/yumi/blob/master/yumi_description/urdf/yumi.urdf
- https://github.com/ankurhanda/robot-assets
"""
from __future__ import annotations

import time
from pathlib import Path
from threading import Thread
from typing import List

import numpy as onp
import tyro
import viser
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from viser.extras import ViserUrdf

from robo_cam_calib.utils.kinova_utils import DeviceConnection


class RobotController:
    def __init__(self):
        self.connection = None
        self.base_cyclic = None

    def connect_to_robot(self):
        device_connection = DeviceConnection(DeviceConnection.IP, port=DeviceConnection.TCP_PORT, credentials=(DeviceConnection.USERNAME, DeviceConnection.PASSWORD))
        router = device_connection.connect()
        self.base_cyclic = BaseCyclicClient(router)

    def get_newest_joint_angles(self) -> List[float]:
        if not self.base_cyclic:
            raise Exception("Not connected to the robot")
        feedback = self.base_cyclic.RefreshFeedback()
        return [actuator.position for actuator in feedback.actuators]

    def disconnect_from_robot(self):
        if self.connection:
            self.connection.disconnect()
            self.connection = None

def update_joint_angles(
        urdf: ViserUrdf, 
        gui_joints: List[viser.GuiInputHandle[float]], 
        robot_controller: RobotController
    ):
    newest_angles = robot_controller.get_newest_joint_angles()
    # convert from degrees to radians
    newest_angles = [angle * onp.pi / 180 for angle in newest_angles]
    # add 0 to the end of the list to make it 8 elements
    newest_angles.append(0)

    for gui, new_angle in zip(gui_joints, newest_angles):
        gui.value = new_angle
    urdf.update_cfg(onp.array(newest_angles))

def continuous_update(urdf: ViserUrdf, gui_joints: List[viser.GuiInputHandle[float]], frequency: float, robot_controller: RobotController):
    interval = 1.0 / frequency
    while True:
        update_joint_angles(urdf, gui_joints, robot_controller)
        time.sleep(interval)

def main(urdf_path: Path) -> None:
    robot_controller = RobotController()
    robot_controller.connect_to_robot()

    server = viser.ViserServer()

    # Create a helper for adding URDFs to Viser. This just adds meshes to the scene,
    # helps us set the joint angles, etc.
    urdf = ViserUrdf(server, urdf_path)

    # Create joint angle sliders.
    gui_joints: List[viser.GuiInputHandle[float]] = []
    initial_angles: List[float] = []
    for joint_name, (lower, upper) in urdf.get_actuated_joint_limits().items():
        lower = lower if lower is not None else -onp.pi
        upper = upper if upper is not None else onp.pi

        initial_angle = 0.0 if lower < 0 and upper > 0 else (lower + upper) / 2.0
        slider = server.add_gui_slider(
            label=joint_name,
            min=lower,
            max=upper,
            step=1e-3,
            initial_value=initial_angle,
        )
        slider.on_update(  # When sliders move, we update the URDF configuration.
            lambda _: urdf.update_cfg(onp.array([gui.value for gui in gui_joints]))
        )

        gui_joints.append(slider)
        initial_angles.append(initial_angle)

    # Create joint reset button.
    reset_button = server.add_gui_button("Reset")

    @reset_button.on_click
    def _(_):
        for g, initial_angle in zip(gui_joints, initial_angles):
            g.value = initial_angle

    # Apply initial joint angles.
    urdf.update_cfg(onp.array([gui.value for gui in gui_joints]))

    # Start continuous update thread at 60Hz
    updater_thread = Thread(target=continuous_update, args=(urdf, gui_joints, 60.0, robot_controller), daemon=True)
    updater_thread.start()

    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    tyro.cli(main)