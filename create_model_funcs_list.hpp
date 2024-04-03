#pragma once
#include <openvino/openvino.hpp>
// Create the model that will trigger the transformation "EliminateReshape" to change the input tensor name
std::shared_ptr<ov::Model> create_model_tensor_name_changed(const std::string save_path = "");

// Create the basic normal model
std::shared_ptr<ov::Model> create_simple_model(const std::string save_path = "");

//Create the dynamic model
std::shared_ptr<ov::Model> create_dynamic_model(const std::string save_path = "");

//Create the dynamic output model
std::shared_ptr<ov::Model> create_dynamic_output_model();