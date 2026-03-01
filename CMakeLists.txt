cmake_minimum_required(VERSION 3.0.2)
project(embodied_ai)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  rospy
  std_msgs
)

add_message_files(
  FILES
    ActuatorCommand.msg
    ActuatorState.msg
)

generate_messages(
  DEPENDENCIES
    std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime rospy std_msgs
)

catkin_install_python(
  PROGRAMS scripts/llm_chat_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
