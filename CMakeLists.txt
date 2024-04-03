cmake_minimum_required(VERSION 3.16)
project(hello_openvino)
set(SOURCE_FILES_HELLO_WORLD
    hello_world.cpp
    create_model_funcs_list.hpp
    create_model_funcs_list.cpp
)

set(SOURCE_MULTI_INFER_FILES
    multi_threads_infer_on_multi_device.cpp
)

set(SOURCE_SINGLE_INFER_FILES
    hello_single_infer.cpp
)

set(SOURCE_AUTO_TEST
    auto_test.cpp
)
set(SOURCE_PURE_METHOD_CALL
    triger_call_pure_method_call.cpp
)
find_package(OpenVINO REQUIRED)
add_executable(hello_openvino ${SOURCE_FILES_HELLO_WORLD})
add_executable(hello_multi_infer ${SOURCE_MULTI_INFER_FILES})
add_executable(hello_single_infer ${SOURCE_SINGLE_INFER_FILES})
add_executable(auto_test ${SOURCE_AUTO_TEST})
add_executable(pure_virtual_call ${SOURCE_PURE_METHOD_CALL})
#add_compile_options(-fsanitize=address)
target_link_libraries(hello_openvino PRIVATE openvino::runtime)
target_link_libraries(hello_multi_infer PRIVATE openvino::runtime pthread)
target_link_libraries(hello_single_infer PRIVATE openvino::runtime pthread)
target_link_libraries(auto_test PRIVATE openvino::runtime pthread)
target_link_libraries(pure_virtual_call PRIVATE pthread)