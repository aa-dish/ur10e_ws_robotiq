"""Starts the AI node (vision + LLM + pick/place).  The Gazebo launch file includes this by default.

  ros2 launch bin_picking_ai bin_picking_ai.launch.py use_llm:=true ollama_model:=llama3.2
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument('use_llm', default_value='true', description='use Ollama to understand commands'),
        DeclareLaunchArgument('ollama_model', default_value='auto', description="'auto' = pick an installed model automatically, or e.g. llama3.2"),
        DeclareLaunchArgument('ollama_host', default_value='http://localhost:11434'),
        DeclareLaunchArgument('left_right', default_value='image', description="'image' or 'robot' viewpoint for left/right"),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
    ]
    node = Node(
        package='bin_picking_ai', executable='bin_picking_node', name='bin_picking', output='screen',
        parameters=[{
            'use_llm': LaunchConfiguration('use_llm'),
            'ollama_model': LaunchConfiguration('ollama_model'),
            'ollama_host': LaunchConfiguration('ollama_host'),
            'left_right': LaunchConfiguration('left_right'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }])
    return LaunchDescription(args + [node])
