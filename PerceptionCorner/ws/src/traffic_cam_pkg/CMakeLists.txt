cmake_minimum_required(VERSION 3.5)
project(traffic_cam_pkg)

find_package(ament_cmake REQUIRED)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# Install config files
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
