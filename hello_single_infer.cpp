// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// clang-format off
#include "openvino/openvino.hpp"
#include "create_model_funcs_list.hpp"

// clang-format on

/**
 * @brief Main with support Unicode paths, wide strings
 */
int main(int argc, char *argv[]) {
  try {
    // -------- Get OpenVINO runtime version --------
    std::cout << ov::get_openvino_version() << std::endl;

    // -------- Parsing and validation of input arguments --------
    if (argc != 3) {
      std::cout << "Usage : " << argv[0]
                 << " <device name> <model path>"
                 << std::endl;
      return EXIT_FAILURE;
    }

    const std::string model_path(argv[2]);
    const std::string device_name(argv[1]);

    // -------- Step 1. Initialize OpenVINO Runtime Core --------
    ov::Core core;

    // -------- Step 2. Read a model --------
    std::cout << "Loading model files: " << model_path << std::endl;
    // std::shared_ptr<ov::Model> model = core.read_model(model_path);
    std::shared_ptr<ov::Model> model = create_dynamic_model();


    OPENVINO_ASSERT(model->inputs().size() == 1,
                    "Sample supports models with 1 input only");

    // -------- Step 3. Set up input
    auto input = model->get_parameters().at(0);
    ov::element::Type input_type = input->get_element_type();
    ov::Shape input_shape = ov::Shape{1, 2};

    // just wrap image data by ov::Tensor without allocating of new memory
    ov::Tensor input_tensor = ov::Tensor{input_type, input_shape};
    float *in_ptr = static_cast<float *>(input_tensor.data());
    if (in_ptr) {
      for (size_t i = 0; i < input_tensor.get_size(); i++) {
        if (i % 2 == 0)
          in_ptr[i] = -1.5;
        else
          in_ptr[i] = 0.5;
      }
    }

    // -------- Step 5. Loading a model to the device --------
    ov::CompiledModel compiled_model = core.compile_model(model, device_name);
    // -------- Step 4. Set up output tensor
    auto output = compiled_model.outputs().at(0);
    ov::element::Type output_type = output.get_element_type();
    ov::Tensor output_tensor = ov::Tensor{output_type, ov::Shape{0}};
    std::cout << "Output data before infer: " << std::hex << output_tensor.data() << std::endl;
    // -------- Step 6. Create an infer request --------
    ov::InferRequest infer_request = compiled_model.create_infer_request();
    // -------- Step 7. Prepare input --------
    infer_request.set_tensor(input, input_tensor);
    infer_request.set_output_tensor(output_tensor);
    // infer_request.set_tensor(output, output_tensor);
    // -------- Step 8. Do inference synchronously --------
    infer_request.infer();
    //infer_request.start_async();
    infer_request.wait();
    // -------- Step 9. Process output

    const ov::Tensor &result_tensor = infer_request.get_output_tensor();
    // Print classification results
    const float *out_ptr = static_cast<float *>(result_tensor.data());
    if (out_ptr) {
      std::cout << "Output data after infer: " << std::hex << out_ptr << std::endl;
      for (int i = 0; i < result_tensor.get_size(); i++) {
        std::cout << out_ptr[i] << " ";
      }
    }
    std::cout << std::endl;
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}