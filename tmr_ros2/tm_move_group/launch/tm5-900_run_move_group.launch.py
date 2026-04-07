import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    args = []
    if len(sys.argv) >= 5:
        for i in range(4, len(sys.argv)):
            args.append(sys.argv[i])

    # Build MoveIt config using the Jazzy-compatible builder
    moveit_config = (
        MoveItConfigsBuilder("tm5-900", package_name="tm5-900_moveit_config")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )

    # move_group node
    # moveit_manage_controllers must be False: tm_driver provides the
    # FollowJointTrajectory action server directly without a controller_manager
    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            moveit_config.to_dict(),
            {'moveit_manage_controllers': False},
            {'trajectory_execution.allowed_execution_duration_scaling': 1.2},
            {'trajectory_execution.allowed_goal_duration_margin': 0.5},
            {'trajectory_execution.allowed_start_tolerance': 0.1},
            {'publish_planning_scene': True},
            {'publish_geometry_updates': True},
            {'publish_state_updates': True},
            {'publish_transforms_updates': True},
        ],
    )

    # RViz
    rviz_config_file = os.path.join(
        get_package_share_directory('tm_move_group'), 'launch', 'run_move_group.rviz'
    )
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
        arguments=['-d', rviz_config_file],
        parameters=[moveit_config.to_dict()],
        additional_env={'LIBGL_ALWAYS_SOFTWARE': '1', 'QT_QPA_PLATFORM': 'xcb'},
    )

    # Static TF: world -> base
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        output='log',
        arguments=['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', 'world', 'base'],
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[moveit_config.robot_description],
    )

    # TM5-900 hardware driver (provides FollowJointTrajectory action server)
    tm_driver_node = Node(
        package='tm_driver',
        executable='tm_driver',
        output='screen',
        arguments=args,
    )

    return LaunchDescription([
        tm_driver_node,
        static_tf,
        robot_state_publisher,
        move_group_node,
        rviz_node,
    ])
