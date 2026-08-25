cmake_minimum_required(VERSION 3.20)
project(autoaim_research LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d dnn)
find_package(DaedalusSimSdk 1 REQUIRED CONFIG)

get_filename_component(AUTOAIM_WORKSPACE_ROOT
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
set(ONNXRUNTIME_ROOT
  "${AUTOAIM_WORKSPACE_ROOT}/deps/onnxruntime-linux-x64-1.22.1"
  CACHE PATH "Pinned ONNX Runtime 1.22.1 installation")
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  PATHS "${ONNXRUNTIME_ROOT}/include"
  NO_DEFAULT_PATH)
find_library(ONNXRUNTIME_LIBRARY onnxruntime
  PATHS "${ONNXRUNTIME_ROOT}/lib"
  NO_DEFAULT_PATH)
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR
    "ONNX Runtime 1.22.1 was not found under ONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT}")
endif()

add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
)

set(TONGJI_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tongji_sp_vision_25)

add_library(tongji_sp_vision_25_core STATIC
  ${TONGJI_ROOT}/tasks/auto_aim/armor.cpp
  ${TONGJI_ROOT}/tasks/auto_aim/target.cpp
  ${TONGJI_ROOT}/tools/extended_kalman_filter.cpp
  ${TONGJI_ROOT}/tools/math_tools.cpp
)
target_include_directories(tongji_sp_vision_25_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/compat
  ${TONGJI_ROOT}
  ${TONGJI_ROOT}/tasks/auto_aim
)
target_link_libraries(tongji_sp_vision_25_core PUBLIC Eigen3::Eigen ${OpenCV_LIBS})

add_library(autoaim_research_core STATIC
  src/coordinates.cpp
  src/detector.cpp
  src/pose_solver.cpp
  src/research_tracker.cpp
)
target_include_directories(autoaim_research_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(autoaim_research_core PUBLIC
  tongji_sp_vision_25_core
  Eigen3::Eigen
  onnxruntime
  ${OpenCV_LIBS}
)

add_executable(autoaim_research_runner src/main.cpp)
target_link_libraries(autoaim_research_runner PRIVATE
  autoaim_research_core
  DaedalusSimSdk::DaedalusSimSdk
)

add_executable(autoaim_research_truth_gimbal src/truth_gimbal_controller.cpp)
target_link_libraries(autoaim_research_truth_gimbal PRIVATE
  Eigen3::Eigen
  DaedalusSimSdk::DaedalusSimSdk
)

include(CTest)
if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  add_executable(autoaim_research_core_test tests/core_test.cpp)
  target_link_libraries(autoaim_research_core_test PRIVATE autoaim_research_core)
  add_test(NAME autoaim_research_core_test COMMAND autoaim_research_core_test)

  add_executable(autoaim_research_model_smoke tests/model_smoke.cpp)
  target_link_libraries(autoaim_research_model_smoke PRIVATE autoaim_research_core)
  add_test(NAME autoaim_research_model_smoke
    COMMAND autoaim_research_model_smoke
      "${AUTOAIM_WORKSPACE_ROOT}/models/engines/windows/armor-0526-fp32-converted.onnx")

  add_test(NAME autoaim_research_lock_test
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify-implementation-lock.py"
      --workspace "${AUTOAIM_WORKSPACE_ROOT}")
endif()
