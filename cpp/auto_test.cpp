#include <future>
#include <iostream>
#include <string>

#include "openvino/op/util/variable.hpp"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset11.hpp"
int main(int argc, char* argv[]) {
    ov::Core core;
    std::cout << ov::get_openvino_version().description << ':' << ov::get_openvino_version().buildNumber << std::endl;
    //core.set_property("AUTO", {{"DEVICE_BIND_BUFFER", "TEST"}});
    //core.get_property("AUTO", "DEVICE_BIND_BUFFER");
    return 0;
}