"""What the orchestrator needs from a robot.  Implemented by the ROS adapter (nodes/ros_robot.py)
and by the offline emulator (sim_robot.py) so the whole logic can be tested without Gazebo."""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .motion.plan import Step


class RobotInterface:
    def get_arm_q(self) -> np.ndarray:                                   # 6 joint angles
        raise NotImplementedError

    def execute_arm(self, step: Step, should_abort: Callable[[], bool]) -> bool:
        raise NotImplementedError

    def execute_gripper(self, theta: float, duration: float, should_abort: Callable[[], bool]) -> bool:
        raise NotImplementedError

    def attach(self, entity: str) -> None:                               # virtual grasp on
        raise NotImplementedError

    def detach(self, entity: str) -> None:                               # virtual grasp off
        raise NotImplementedError

    def sleep(self, seconds: float) -> None:
        raise NotImplementedError

    def capture(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fresh (rgb HxWx3 uint8, depth HxW float32 [m], K 3x3, T_base_cam 4x4)."""
        raise NotImplementedError

    def entity_poses(self) -> Optional[Dict[str, np.ndarray]]:
        """Ground-truth xyz (base_link frame) of simulator entities, or None if not available."""
        return None

    def say(self, message: str) -> None:
        print(message)
