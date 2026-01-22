import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    pkg_dir = get_package_share_directory('traffic_cam_pkg')
    
    # 1. Image Encoder (Pre-processor)
    # TrafficCamNet expects 960x544 and 1/255 scale (0.0 to 1.0)
    encoder_node = ComposableNode(
    package='isaac_ros_dnn_image_encoder',
    plugin='nvidia::isaac_ros::dnn_inference::DnnImageEncoderNode',
    name='dnn_image_encoder',
    parameters=[{
        'input_image_width': 1280,    # ADD THIS: Match your ZED resolution
        'input_image_height': 720,   # ADD THIS: Match your ZED resolution
        'network_image_width': 960,
        'network_image_height': 544,
        'image_mean': [0.0, 0.0, 0.0],
        'image_stddev': [1.0, 1.0, 1.0],
    }],
    remappings=[('image', '/zed/zed_node/left/image_rect_color')]
    )

    # 2. TensorRT Node
    tensorrt_node = ComposableNode(
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRtNode',
        name='tensor_rt',
        parameters=[{
            'model_file_path': '/home/nourstalgie/FYP-Autonomous-Ground-Vehicles/PerceptionCorner/ws/models/trafficcamnet/resnet18_trafficcamnet_pruned.onnx',
            'engine_file_path': '/home/nourstalgie/FYP-Autonomous-Ground-Vehicles/PerceptionCorner/ws/models/trafficcamnet/resnet18_trafficcamnet_pruned.onnx_b1_gpu0_fp16.engine',
            'input_tensor_names': ['input_1'],
            'output_tensor_names': ['output_bbox/BiasAdd', 'output_cov/Sigmoid'],
            'force_engine_update': False
        }]
    )

    # 3. DetectNet Decoder
    decoder_node = ComposableNode(
        package='isaac_ros_detectnet',
        plugin='nvidia::isaac_ros::detectnet::DetectNetDecoderNode',
        name='detectnet_decoder',
        parameters=[{
            'label_list': ['car', 'person', 'road_sign', 'bicycle'],
            'confidence_threshold': 0.3
        }]
    )

    return LaunchDescription([
        ComposableNodeContainer(
            name='traffic_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=[encoder_node, tensorrt_node, decoder_node],
            output='screen'
        )
    ])
