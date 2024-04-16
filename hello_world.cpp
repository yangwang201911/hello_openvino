#include <iostream>
#include <openvino/openvino.hpp>
#include <openvino/op/ops.hpp>
#include "create_model_funcs_list.hpp"
using namespace std;
int main(int argc, char* argv[]) {
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(argv[1]);
    ov::CompiledModel compiled_model = core.compile_model(model, "GPU");
    auto outputShape = compiled_model.outputs().at(0).get_shape();
    std::cout << "output shape after compilation: " << outputShape << std::endl;
    return 0;
}