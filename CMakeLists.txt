cmake_minimum_required(VERSION 3.16)
project(hello_openvino)
set(SOURCE_FILES_HELLO_WORLD
    cpp/hello_world.cpp
    cpp/create_model_funcs_list.hpp
    cpp/create_model_funcs_list.cpp
)

set(SOURCE_MULTI_INFER_FILES
    cpp/multi_threads_infer_on_multi_device.cpp
)

set(SOURCE_SINGLE_INFER_FILES
    cpp/hello_single_infer.cpp
    cpp/create_model_funcs_list.cpp
)

set(SOURCE_AUTO_TEST
    cpp/auto_test.cpp
)
find_package(OpenVINO REQUIRED)
add_executable(hello_world ${SOURCE_FILES_HELLO_WORLD})
add_executable(hello_multi_infer ${SOURCE_MULTI_INFER_FILES})
add_executable(hello_single_infer ${SOURCE_SINGLE_INFER_FILES})
add_executable(auto_test ${SOURCE_AUTO_TEST})
#add_compile_options(-fsanitize=address)
target_link_libraries(hello_world PRIVATE openvino::runtime)
target_link_libraries(hello_multi_infer PRIVATE openvino::runtime pthread)
target_link_libraries(hello_single_infer PRIVATE openvino::runtime pthread)
target_link_libraries(auto_test PRIVATE openvino::runtime pthread)