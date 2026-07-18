# 文件路径: robomaster2025/src/RobotEstimator/tinympc/CMakeLists.txt

# 查找 Eigen3 (必须)
find_package(Eigen3 REQUIRED)

add_library(tinympcstatic STATIC
    admm.cpp
    tiny_api.cpp
    codegen.cpp
    rho_benchmark.cpp
)

set_property(TARGET tinympcstatic PROPERTY POSITION_INDEPENDENT_CODE ON)

# 包含当前目录和上级目录 (以便引用 tinympc 内部头文件)
target_include_directories(tinympcstatic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

# [重点修复] 链接 Eigen3 库，这样会自动处理头文件路径
target_link_libraries(tinympcstatic PUBLIC Eigen3::Eigen)

if(USING_CODEGEN)
    # Files that are needed for embedded code generation
    list( APPEND EMBEDDED_FILES
          "${CMAKE_CURRENT_SOURCE_DIR}/admm.cpp"
          "${CMAKE_CURRENT_SOURCE_DIR}/admm.hpp"
          "${CMAKE_CURRENT_SOURCE_DIR}/tiny_api.cpp"
          "${CMAKE_CURRENT_SOURCE_DIR}/tiny_api.hpp"
          "${CMAKE_CURRENT_SOURCE_DIR}/types.hpp"
          "${CMAKE_CURRENT_SOURCE_DIR}/tiny_api_constants.hpp" )

    foreach( f ${EMBEDDED_FILES} )
        get_filename_component( fname ${f} NAME )
        set( dest_file "${EMBEDDED_BUILD_TINYMPC_DIR}/${fname}" )
        list( APPEND EMBEDDED_BUILD_TINYMPC_FILES "${dest_file}" )

        add_custom_command(OUTPUT ${dest_file}
                            COMMAND ${CMAKE_COMMAND} -E copy "${f}" "${dest_file}"
                            DEPENDS ${f}
                            COMMENT "Copying ${fname}")
    endforeach()

    add_custom_target( copy_codegen_tinympc_files DEPENDS ${EMBEDDED_BUILD_TINYMPC_FILES} )
    add_dependencies( copy_codegen_files copy_codegen_tinympc_files )
endif(USING_CODEGEN)
