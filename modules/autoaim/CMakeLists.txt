cmake_minimum_required(VERSION 3.16)
project(auto_aim_armor_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(AIM_SIM_WITH_ROS2 "Build the ROS2 simulator adapter node" ON)
option(AIM_SIM_WITH_VIVSIONN_TRT "Build vivsionn TensorRT-backed detector pipelines" OFF)
option(AIM_SIM_BUILD_MPC_SMOKE "Build a small smoke executable for the copied MPC module" ON)
option(AIM_SIM_BUILD_CONTROL_LOOP_TESTS "Build deterministic fixed-rate control loop tests" ON)
option(AIM_SIM_BUILD_PNP_CANDIDATE_TESTS "Build deterministic planar PnP candidate tests" ON)
option(AIM_SIM_BUILD_TCP_IMAGE_TESTS "Build deterministic raw TCP image receiver tests" ON)
option(AIM_SIM_BUILD_ARMOR_ASYNC_SCHEDULER_TESTS "Build deterministic armor async scheduler tests" ON)
option(AIM_SIM_BUILD_SCENE_CONTROL_CLI "Build the SDK-only dynamic shooting-range control CLI" ON)
option(AIM_SIM_TCP_IMAGE_TEST_SANITIZERS "Enable ASan/UBSan for raw TCP receiver tests" OFF)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d dnn imgcodecs)
find_package(DaedalusSimSdk 1 REQUIRED CONFIG)

set(AIM_SIM_PUBLIC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(AIM_SIM_VIVSIONN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/aim_core_from_vivsionn)

add_subdirectory(
  ${AIM_SIM_VIVSIONN_DIR}/RobotEstimator/tinympc
  ${CMAKE_CURRENT_BINARY_DIR}/tinympc
)

add_library(aim_vivsionn_mpc STATIC
  ${AIM_SIM_VIVSIONN_DIR}/FireControl/second_order_position_mpc.cpp
)
target_include_directories(aim_vivsionn_mpc PUBLIC
  ${AIM_SIM_VIVSIONN_DIR}/FireControl
  ${AIM_SIM_VIVSIONN_DIR}/RobotEstimator
)
target_link_libraries(aim_vivsionn_mpc PUBLIC Eigen3::Eigen tinympcstatic)

add_library(aim_sim_fixed_rate_control STATIC
  src/sim_adapter/fixed_rate_command_loop.cpp
)
target_include_directories(aim_sim_fixed_rate_control PUBLIC
  ${AIM_SIM_PUBLIC_INCLUDE_DIR}
)

add_library(aim_sim_auto_aim_armor_core STATIC
  src/sim_adapter/sim_command.cpp
)

if(AIM_SIM_WITH_VIVSIONN_TRT)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 72 86 89 CACHE STRING
      "CUDA architectures for Jetson NX/Xavier, Ampere, and Ada targets")
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  find_package(yaml-cpp REQUIRED)

  find_library(TENSORRT_NVINFER nvinfer REQUIRED)
  find_library(TENSORRT_NVINFER_PLUGIN nvinfer_plugin REQUIRED)

  target_sources(aim_sim_auto_aim_armor_core PRIVATE
    src/aim_core_bridge/vivsionn_pipeline.cpp
    src/aim_core_bridge/stage3_capture.cpp
    ${AIM_SIM_VIVSIONN_DIR}/AngleSolver/AngleSolver.cpp
    ${AIM_SIM_VIVSIONN_DIR}/AngleSolver/opencv_extended.cpp
    ${AIM_SIM_VIVSIONN_DIR}/ArmorDetector/mt_detector_tensorrt.cpp
    ${AIM_SIM_VIVSIONN_DIR}/ArmorDetector/preprocess_kernel.cu
    ${AIM_SIM_VIVSIONN_DIR}/FireControl/firecontrol.cpp
    ${AIM_SIM_VIVSIONN_DIR}/RobotEstimator/extendedkalmanfilter.cpp
    ${AIM_SIM_VIVSIONN_DIR}/RobotEstimator/planner.cpp
    ${AIM_SIM_VIVSIONN_DIR}/RobotEstimator/robotestimator.cpp
    ${AIM_SIM_VIVSIONN_DIR}/RobotEstimator/target.cpp
    ${AIM_SIM_VIVSIONN_DIR}/RobotEstimator/ypd_angle_tracker.cpp
    ${AIM_SIM_VIVSIONN_DIR}/tools/extended_kalman_filter.cpp
    ${AIM_SIM_VIVSIONN_DIR}/tools/math_tools.cpp
    ${AIM_SIM_VIVSIONN_DIR}/tools/trajectory.cpp
    ${AIM_SIM_VIVSIONN_DIR}/tools/Timer/timer.cpp
  )
  target_compile_definitions(aim_sim_auto_aim_armor_core PUBLIC AIM_SIM_WITH_VIVSIONN_TRT=1 AIM_SIM_FEATURE_ARMOR=1 AIM_SIM_FEATURE_BUFF=0)
  target_link_libraries(aim_sim_auto_aim_armor_core PUBLIC
    yaml-cpp
    CUDA::cudart
    CUDA::nppc
    CUDA::nppig
    CUDA::nppim
    ${TENSORRT_NVINFER}
    ${TENSORRT_NVINFER_PLUGIN}
    aim_vivsionn_mpc
  )
else()
  target_sources(aim_sim_auto_aim_armor_core PRIVATE src/aim_core_bridge/noop_pipeline.cpp)
  target_compile_definitions(aim_sim_auto_aim_armor_core PUBLIC AIM_SIM_WITH_VIVSIONN_TRT=0 AIM_SIM_FEATURE_ARMOR=1 AIM_SIM_FEATURE_BUFF=0)
endif()

target_include_directories(aim_sim_auto_aim_armor_core PUBLIC
  ${AIM_SIM_PUBLIC_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/src/aim_core_bridge/stubs
  ${AIM_SIM_VIVSIONN_DIR}
  ${AIM_SIM_VIVSIONN_DIR}/AngleSolver
  ${AIM_SIM_VIVSIONN_DIR}/ArmorDetector
  ${AIM_SIM_VIVSIONN_DIR}/ConcurrentQueue
  ${AIM_SIM_VIVSIONN_DIR}/FireControl
  ${AIM_SIM_VIVSIONN_DIR}/GeneralDeclaration
  ${AIM_SIM_VIVSIONN_DIR}/RobotEstimator
  ${AIM_SIM_VIVSIONN_DIR}/parameters
  ${AIM_SIM_VIVSIONN_DIR}/tools
  ${AIM_SIM_VIVSIONN_DIR}/tools/Timer
)
target_link_libraries(aim_sim_auto_aim_armor_core PUBLIC
  Eigen3::Eigen
  ${OpenCV_LIBS}
  aim_sim_fixed_rate_control
)

if(AIM_SIM_BUILD_CONTROL_LOOP_TESTS)
  enable_testing()
  add_executable(aim_fixed_rate_command_loop_test
    src/tests/fixed_rate_command_loop_test.cpp
  )
  target_link_libraries(aim_fixed_rate_command_loop_test PRIVATE
    aim_sim_fixed_rate_control
  )
  add_test(NAME aim_fixed_rate_command_loop_test COMMAND aim_fixed_rate_command_loop_test)
endif()

if(AIM_SIM_BUILD_ARMOR_ASYNC_SCHEDULER_TESTS)
  find_package(Threads REQUIRED)
  enable_testing()
  add_executable(aim_armor_async_scheduler_test
    src/tests/armor_async_scheduler_test.cpp
  )
  target_include_directories(aim_armor_async_scheduler_test PRIVATE
    ${AIM_SIM_PUBLIC_INCLUDE_DIR}
    ${AIM_SIM_VIVSIONN_DIR}/ArmorDetector
    ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(aim_armor_async_scheduler_test PRIVATE Threads::Threads)
  add_test(NAME aim_armor_async_scheduler_test COMMAND aim_armor_async_scheduler_test)
endif()

if(UNIX)
  find_package(Threads REQUIRED)
  add_library(aim_sim_tcp_image_receiver STATIC
    src/sim_adapter/tcp_image_receiver.cpp
  )
  target_include_directories(aim_sim_tcp_image_receiver PUBLIC
    ${AIM_SIM_PUBLIC_INCLUDE_DIR}
  )
  target_link_libraries(aim_sim_tcp_image_receiver PUBLIC Threads::Threads)
  target_link_libraries(aim_sim_tcp_image_receiver PUBLIC
    DaedalusSimSdk::DaedalusSimSdk
  )
  target_compile_options(aim_sim_tcp_image_receiver PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Werror
  )

  if(AIM_SIM_TCP_IMAGE_TEST_SANITIZERS)
    target_compile_options(aim_sim_tcp_image_receiver PRIVATE
      -fsanitize=address,undefined -fno-omit-frame-pointer
    )
    target_link_options(aim_sim_tcp_image_receiver PRIVATE
      -fsanitize=address,undefined
    )
  endif()

  if(AIM_SIM_BUILD_TCP_IMAGE_TESTS)
    enable_testing()
    add_executable(aim_tcp_image_receiver_test
      src/tests/tcp_image_receiver_test.cpp
    )
    target_link_libraries(aim_tcp_image_receiver_test PRIVATE
      aim_sim_tcp_image_receiver
    )
    target_compile_options(aim_tcp_image_receiver_test PRIVATE
      -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Werror
    )
    if(AIM_SIM_TCP_IMAGE_TEST_SANITIZERS)
      target_compile_options(aim_tcp_image_receiver_test PRIVATE
        -fsanitize=address,undefined -fno-omit-frame-pointer
      )
      target_link_options(aim_tcp_image_receiver_test PRIVATE
        -fsanitize=address,undefined
      )
    endif()
    add_test(NAME aim_tcp_image_receiver_test COMMAND aim_tcp_image_receiver_test)
  endif()
endif()

if(UNIX)
  add_executable(aim_sim_talos_auto_aim_bridge src/sim_adapter/talos_bridge_node.cpp)
  target_link_libraries(aim_sim_talos_auto_aim_bridge PRIVATE
    aim_sim_auto_aim_armor_core
    aim_sim_tcp_image_receiver
    DaedalusSimSdk::DaedalusSimSdk
  )

  if(AIM_SIM_BUILD_SCENE_CONTROL_CLI)
    add_executable(aim_sim_scene_control_cli
      src/sim_adapter/scene_control_cli.cpp
    )
    target_link_libraries(aim_sim_scene_control_cli PRIVATE
      DaedalusSimSdk::DaedalusSimSdk
    )
  endif()
endif()

if(AIM_SIM_WITH_ROS2)
  find_package(ament_cmake REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(tf2_msgs REQUIRED)
  find_package(rm_interfaces REQUIRED)

  add_executable(aim_sim_auto_aim_node src/sim_adapter/ros2_bridge_node.cpp)
  target_link_libraries(aim_sim_auto_aim_node aim_sim_auto_aim_armor_core)
  ament_target_dependencies(aim_sim_auto_aim_node
    rclcpp
    sensor_msgs
    geometry_msgs
    tf2_msgs
    rm_interfaces
  )

  install(TARGETS aim_sim_auto_aim_node
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )
  install(DIRECTORY config
    DESTINATION share/${PROJECT_NAME}
  )
  ament_package()
endif()

if(AIM_SIM_BUILD_MPC_SMOKE)
  add_executable(aim_mpc_smoke src/tests/mpc_smoke.cpp)
  target_link_libraries(aim_mpc_smoke PRIVATE aim_vivsionn_mpc)
endif()

if(AIM_SIM_WITH_VIVSIONN_TRT)
  if(AIM_SIM_BUILD_PNP_CANDIDATE_TESTS)
    enable_testing()
    add_executable(aim_angle_solver_pnp_candidates_test
      src/tests/angle_solver_pnp_candidates_test.cpp
    )
    target_link_libraries(aim_angle_solver_pnp_candidates_test PRIVATE aim_sim_auto_aim_armor_core)
    add_test(
      NAME aim_angle_solver_pnp_candidates_test
      COMMAND aim_angle_solver_pnp_candidates_test
    )
    set_tests_properties(aim_angle_solver_pnp_candidates_test PROPERTIES
      ENVIRONMENT "AIM_SIM_PARAM_YAML=${CMAKE_CURRENT_SOURCE_DIR}/config/param.sim.yaml"
    )
  endif()

endif()
