// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
// clang-format off
#include <openvino/openvino.hpp>

// clang-format on

/**
 * @brief Main with support Unicode paths, wide strings
 */
int main(int argc, char* argv[]) {
    int * ptr = (int*) malloc(sizeof(ptr));
    try {
        // -------- Get OpenVINO runtime version --------
        std::cout << ov::get_openvino_version() << std::endl;

        // -------- Parsing and validation of input arguments --------
        if (argc != 2) {
            std::cout << "Usage : " << argv[0] << " <path_to_model>" << std::endl;
            return EXIT_FAILURE;
        }

        const std::string args(argv[0]);
        const std::string model_path(argv[1]);

        // -------- Step 1. Initialize OpenVINO Runtime Core --------
        ov::Core core;

        // -------- Step 2. Read a model --------aaa
        std::cout << "Loading model files: " << model_path << std::endl;
        std::shared_ptr<ov::Model> model = core.read_model(model_path);
        // printInputAndOutputsInfo(*model);

        // OPENVINO_ASSERT(model->inputs().size() == 1, "Sample supports models with 1 input only");

        // -------- Step 3. Set up input
        // fill Inputs
        std::vector<ov::Tensor> inputData;
        const auto& functionParams = model->get_parameters();
        ov::CompiledModel compiled_model_1 = core.compile_model(model, "CPU", ov::hint::performance_mode("THROUGHPUT"));
        ov::CompiledModel compiled_model_2 =
            core.compile_model(model, "GPU", ov::hint::performance_mode("THROUGHPUT"));
        auto infer_request_num_1 = compiled_model_1.get_property(ov::optimal_number_of_infer_requests);
        auto infer_request_num_2 = compiled_model_2.get_property(ov::optimal_number_of_infer_requests);
        for (size_t j = 0; j < infer_request_num_1 + infer_request_num_2; j++) {
            for (size_t i = 0; i < functionParams.size(); i++) {
                const auto& param = functionParams[i];

                ov::Tensor blob;
                if (param->get_partial_shape().is_static()) {
                    blob = ov::Tensor(param->get_element_type(), param->get_shape());
                } else {
                    throw("only test static models");
                }
                float* in_ptr = static_cast<float*>(blob.data());
                for (size_t i = 0; i < blob.get_size(); i++) {
                    in_ptr[i] = 1.0;
                }
                inputData.push_back(blob);
            }
        }
        std::cout << "inputData size is " << inputData.size() << std::endl;
        // -------- Step 5. Loading a model to the device --------
        // ov::CompiledModel compiled_model_1 = core.compile_model(model, device_name,
        // ov::hint::performance_mode("THROUGHPUT")); auto execGraphInfo = compiled_model.get_runtime_model();
        // ov::serialize(execGraphInfo, "/home/bell/gpu_exe_model.xml", "/home/bell/model.bin");
        std::vector<ov::InferRequest> infer_requests;
        for (size_t j = 0; j < infer_request_num_1; j++) {
            ov::InferRequest infer_request = compiled_model_1.create_infer_request();
            infer_requests.push_back(infer_request);
        }
        for (size_t j = 0; j < infer_request_num_2; j++) {
            ov::InferRequest infer_request = compiled_model_2.create_infer_request();
            infer_requests.push_back(infer_request);
        }
        // -------- Step 6. Create an infer request --------
        std::function<void(int)> func = [&](int index) {
            for (size_t i = 0; i < functionParams.size(); ++i) {
                infer_requests[index].set_tensor(compiled_model_1.input(i), inputData[index * 3 + i]);
            }
            infer_requests[index].start_async();
            infer_requests[index].wait();
        };
        for (int j = 0; j < 10000; j++) {
            if (j % 1000 == 0)
                std::cout << "iteration " << j << "th" << std::endl;
            std::vector<std::thread> threads;
            for (int i = 0; i < infer_request_num_1 + infer_request_num_2; i++)
                threads.push_back(std::thread(func, i));
            for (auto& iter : threads) {
                if (iter.joinable())
                    iter.join();
            }
        }
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
