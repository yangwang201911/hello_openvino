#include <iostream>
#include <openvino/openvino.hpp>
#include <openvino/op/ops.hpp>
#include "create_model_funcs_list.hpp"
using namespace std;
int main(int argc, char* argv[]) {
    ov::Core core;
    auto model = create_simple_model();

    auto qureyModel = core.query_model(model, "AUTO.110");
    return 0;
}