import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    bringup_dir = get_package_share_directory('seb_naver_bringup')
    sim_dir = get_package_share_directory('seb_naver_simulation')
    ta_dir = get_package_share_directory('terrain_analyzer')
    nav2_dir = get_package_share_directory('nav2_bringup')
    
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    params_file = os.path.join(bringup_dir, 'params', 'nav2_params.yaml')

    # 1. Gazebo Simulation
    sim_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(sim_dir, 'launch', 'sim.launch.py'))
    )

    # 2. Terrain Analyzer
    ta_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ta_dir, 'launch', 'terrain_analyzer.launch.py')),
        # In a real scenario we might pass specific arguments here
    )

    # 3. SE2Grid to OccupancyGrid Converter
    grid_converter_cmd = Node(
        package='seb_naver_bringup',
        executable='se2_to_occupancy_node',
        name='se2_to_occupancy_node',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'input_topic': '/terrain_analyzer/fused_map'}, # Assuming this is the TA output topic
            {'output_topic': '/map'},
            {'risk_threshold': 1.0},
            {'layer_name': 'risk'}
        ]
    )

    # 4. Nav2
    nav2_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_dir, 'launch', 'navigation_launch.py')),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': params_file,
            'autostart': 'true'
        }.items()
    )

    # Static TF for map->odom (since we aren't running SLAM)
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        output='screen'
    )

    # Remap scan_3d to the topic terrain_analyzer expects
    # In WP6 analysis, it says: /scan_3d -> remap to ~/cloud
    # Wait, terrain analyzer subscribes to "cloud". We can remap it in TA launch, or here.
    # Actually, we can just run a relay or remap globally if needed, but it's cleaner to let TA launch handle it or pass it.
    
    return LaunchDescription([
        sim_cmd,
        static_tf,
        ta_cmd,
        grid_converter_cmd,
        nav2_cmd
    ])
