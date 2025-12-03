from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    Command,
    FindExecutable,
    PathJoinSubstitution,
    LaunchConfiguration,
    PythonExpression,
    TextSubstitution,
    IfElseSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    declare_sim_arg = DeclareLaunchArgument(
        name="sim",
        default_value="False",
        description=(
            "Run `ros2 launch neural_controller launch.py sim:=True` to run the robot "
            "in the Mujoco simulator, otherwise the default value of False will run the real robot."
        ),
    )

    xacro_file = PathJoinSubstitution(
        [
            FindPackageShare("pupper_v3_description"),
            "description",
            IfElseSubstitution(
                condition=PythonExpression(LaunchConfiguration("sim")),
                if_value=TextSubstitution(text="pupper_v3_mujoco.urdf.xacro"),
                else_value=TextSubstitution(text="pupper_v3.urdf.xacro"),
            ),
        ]
    )

    robot_description_content = Command(
        [PathJoinSubstitution([FindExecutable(name="xacro")]), " ", xacro_file]
    )
    robot_description = {"robot_description": robot_description_content}

    controller_params = ParameterFile(
        PathJoinSubstitution([FindPackageShare("neural_controller"), "launch", "config.yaml"]),
        allow_substs=True,
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
    )

    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[controller_params],
        output="both",
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
            "--controller-manager-timeout",
            "30",
        ],
        output="both",
    )

    imu_sensor_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "imu_sensor_broadcaster",
            "--controller-manager",
            "/controller_manager",
            "--controller-manager-timeout",
            "30",
        ],
        output="both",
    )

    neural_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "neural_controller",
            "--controller-manager",
            "/controller_manager",
            "--controller-manager-timeout",
            "30",
            "--inactive",
        ],
        output="both",
    )

    nodes = [
        robot_state_publisher,
        control_node,
        joint_state_broadcaster_spawner,
        imu_sensor_broadcaster_spawner,
        neural_controller_spawner,
    ]

    return LaunchDescription([declare_sim_arg, *nodes])

