from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('terrain_analyzer')

    # Launch arguments
    odom_topic_arg = DeclareLaunchArgument(
        'odom_topic', default_value='/odom',
        description='Odometry topic to remap to'
    )
    cloud_topic_arg = DeclareLaunchArgument(
        'cloud_topic', default_value='/scan_3d',
        description='Point cloud topic to remap to'
    )
    params_file_arg = DeclareLaunchArgument(
        'params_file', default_value='elevation_map_gpu.yaml',
        description='Name of the parameter file in the params/ directory'
    )

    odom_topic = LaunchConfiguration('odom_topic')
    cloud_topic = LaunchConfiguration('cloud_topic')
    params_file = LaunchConfiguration('params_file')

    # Terrain analyzer node
    terrain_analyzer_node = Node(
        package='terrain_analyzer',
        executable='terrain_analyzer_node',
        name='terrain_analyzer_node',
        output='screen',
        parameters=[
            os.path.join(pkg_share, 'params', 'elevation_map_gpu.yaml'),
            {'param_file_path': os.path.join(pkg_share, 'params', 'elevation_map_gpu.yaml')}
        ],
        remappings=[
            ('odom', odom_topic),
            ('cloud', cloud_topic),
            ('input_cloud', cloud_topic),
        ],
    )

    return LaunchDescription([
        odom_topic_arg,
        cloud_topic_arg,
        params_file_arg,
        terrain_analyzer_node,
    ])
