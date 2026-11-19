import json
import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    ur_desc_path = get_package_share_directory('ur_description')
    gazebo_pkg_dir = get_package_share_directory('gazebo')
    ai_pkg_dir = get_package_share_directory('bin_picking_ai')
    models_dir = os.path.join(gazebo_pkg_dir, 'models')

    env = {
        'GZ_SIM_SYSTEM_PLUGIN_PATH': ':'.join([
            os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', ''),
            os.path.join(get_package_share_directory('ros_gz_sim'), 'lib')]),
        'IGN_GAZEBO_SYSTEM_PLUGIN_PATH': ':'.join([
            os.environ.get('IGN_GAZEBO_SYSTEM_PLUGIN_PATH', ''),
            os.path.join(get_package_share_directory('ros_gz_sim'), 'lib')]),
        # models/ is added so that  model://<name>  in the world file works on every computer
        'GZ_SIM_RESOURCE_PATH': ':'.join([
            os.environ.get('GZ_SIM_RESOURCE_PATH', ''), os.path.dirname(ur_desc_path), models_dir]),
    }

    urdf_xacro_path = os.path.join(ur_desc_path, 'urdf', 'ur.urdf.xacro')
    urdf_output_path = os.path.join(ur_desc_path, 'urdf', 'generated.urdf')
    urdf_content = xacro.process_file(urdf_xacro_path, mappings={
        'name': 'ur', 'ur_type': 'ur10e', 'tf_prefix': '', 'force_abs_paths': 'true',
        'gripper': os.environ.get('UR_GRIPPER', '2f140').strip().lower()}).toxml()
    os.makedirs(os.path.dirname(urdf_output_path), exist_ok=True)
    with open(urdf_output_path, 'w') as f:
        f.write(urdf_content)

    world_file_path = os.path.join(gazebo_pkg_dir, 'worlds', 'bin_picking_world.sdf')

    entities = sorted({o['name'] for o in json.load(open(os.path.join(ai_pkg_dir, 'data', 'world_objects.json')))})
    bridge_args = [
        '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        '/d435/image@sensor_msgs/msg/Image[gz.msgs.Image',
        '/d435/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
        '/d435/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        '/d435/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
        '/world/empty/pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
    ]
    for e in entities:                                   # ROS -> Gazebo (virtual grasp)
        bridge_args.append(f'/bin_picking/attach/{e}@std_msgs/msg/Empty]gz.msgs.Empty')
        bridge_args.append(f'/bin_picking/detach/{e}@std_msgs/msg/Empty]gz.msgs.Empty')

    use_sim_time = {'use_sim_time': True}

    robot_state_pub_node = Node(package='robot_state_publisher', executable='robot_state_publisher',
                                output='both', parameters=[{'robot_description': urdf_content}, use_sim_time])
    bridge = Node(package='ros_gz_bridge', executable='parameter_bridge', arguments=bridge_args, output='screen')

    def spawner(name):
        return Node(package='controller_manager', executable='spawner',
                    arguments=[name, '--controller-manager', '/controller_manager',
                               '--controller-manager-timeout', '300', '--service-call-timeout', '120'])

    # The DetachableJoint plugin welds every object to the gripper as soon as the robot spawns.
    # A detach request is ignored until the weld exists, so release repeatedly for a while.
    release_cmd = ' ; '.join(
        f'gz topic -t /bin_picking/detach/{e} -m gz.msgs.Empty -p "unused: true"' for e in entities)

    def release_objects():
        return ExecuteProcess(cmd=['bash', '-c', release_cmd], output='log')

    gazebo_process = ExecuteProcess(cmd=['gz', 'sim', '-r', world_file_path], output='screen', additional_env=env)

    spawn_robot_arm = ExecuteProcess(
        cmd=['gz', 'service', '-s', '/world/empty/create', '--reqtype', 'gz.msgs.EntityFactory',
             '--reptype', 'gz.msgs.Boolean', '--timeout', '5000', '--req',
             f'sdf_filename: "{urdf_output_path}", name: "bin_picking_robot", '
             f'pose: {{position: {{x: 0.842175, y: -0.500138, z: 0.6}}}}'],
        output='screen')

    launch_ai = DeclareLaunchArgument('launch_ai', default_value='true',
                                      description='start the vision / LLM pick-and-place node')
    ai = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ai_pkg_dir, 'launch', 'bin_picking_ai.launch.py')),
        condition=IfCondition(LaunchConfiguration('launch_ai')))

    return LaunchDescription([
        launch_ai,
        gazebo_process,
        TimerAction(period=3.0, actions=[robot_state_pub_node, bridge]),
        TimerAction(period=5.0, actions=[spawn_robot_arm]),
        *[TimerAction(period=t, actions=[release_objects()]) for t in (7.0, 8.0, 9.0, 10.0, 12.0, 15.0)],
        TimerAction(period=9.0, actions=[spawner('joint_state_broadcaster')]),
        TimerAction(period=11.0, actions=[spawner('arm_controller')]),
        TimerAction(period=12.0, actions=[spawner('gripper_controller')]),
        TimerAction(period=14.0, actions=[ai]),
    ])
