"""MoveIt 2 move_group for the simulated UR10e.   Start the Gazebo launch first (launch_ai:=false if you only want MoveIt).

  ros2 launch ur10e_moveit_config move_group.launch.py
then open RViz with the MotionPlanning display (fixed frame: base_link) or call the /move_action action.
"""
import os

import xacro
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def _load(pkg, rel):
    with open(os.path.join(get_package_share_directory(pkg), rel)) as f:
        return yaml.safe_load(f)


def generate_launch_description():
    ur = get_package_share_directory('ur_description')
    cfg = get_package_share_directory('ur10e_moveit_config')
    urdf = xacro.process_file(os.path.join(ur, 'urdf', 'ur.urdf.xacro'), mappings={
        'name': 'ur', 'ur_type': 'ur10e', 'tf_prefix': '', 'force_abs_paths': 'true'}).toxml()
    srdf = open(os.path.join(cfg, 'config', 'ur10e.srdf')).read()
    params = {
        'robot_description': urdf,
        'robot_description_semantic': srdf,
        'robot_description_kinematics': _load('ur10e_moveit_config', 'config/kinematics.yaml'),
        'robot_description_planning': _load('ur10e_moveit_config', 'config/joint_limits.yaml'),
        'planning_pipelines': {'pipeline_names': ['ompl']},
        'ompl': _load('ur10e_moveit_config', 'config/ompl_planning.yaml'),
        'publish_robot_description_semantic': True,
        'use_sim_time': True,
        'publish_planning_scene': True,
        'publish_geometry_updates': True,
        'publish_state_updates': True,
        'publish_transforms_updates': True,
    }
    params.update(_load('ur10e_moveit_config', 'config/moveit_controllers.yaml'))
    params['trajectory_execution'] = {'allowed_execution_duration_scaling': 2.0, 'allowed_goal_duration_margin': 1.0,
                                      'allowed_start_tolerance': 0.05}
    move_group = Node(package='moveit_ros_move_group', executable='move_group', output='screen', parameters=[params])
    return LaunchDescription([move_group])
