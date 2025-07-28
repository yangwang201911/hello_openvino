#include <cassert>
#include <iostream>
#include <openvino/openvino.hpp>
#include <random>
#include <vector>

#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/transpose.hpp"

void print_2x2_matrix(const float* matrix, const std::string& name) {
    std::cout << name << " (2x2):" << std::endl;
    std::cout << "[[" << std::setw(8) << std::fixed << std::setprecision(1) 
              << matrix[0] << ", " << std::setw(8) << matrix[1] << "]," << std::endl;
    std::cout << " [" << std::setw(8) << matrix[2] << ", " << std::setw(8) 
              << matrix[3] << "]]" << std::endl;
}

std::shared_ptr<ov::Model> create_2x2_matmul_model() {
    // Create input parameters for 2x2 matrices
    auto input1_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});
    auto input2_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});
    
    // Create MatMul operation (no transpose)
    auto matmul_op = std::make_shared<ov::op::v0::MatMul>(input1_param, input2_param, false, false);

    // Create model (using the same pattern as model_multiply.cpp)
    auto model = std::make_shared<ov::Model>(ov::NodeVector{matmul_op}, 
                                             std::vector<std::shared_ptr<ov::op::v0::Parameter>>{input1_param, input2_param});

    return model;
}

// 随机生成输入数据
std::vector<float> generate_random_data(size_t size) {
    std::vector<float> data(size);
    std::mt19937 gen(42);  // 固定随机种子
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& val : data) {
        val = dist(gen);
    }
    return data;
}

std::shared_ptr<ov::Model> create_model(ov::element::Type& model_type,
                                        const std::vector<int64_t>& target_transpose_order) {
    ov::PartialShape input_a_shape = {-1, -1, -1, -1};
    ov::PartialShape input_b_shape = {-1, -1};

    auto input_a = std::make_shared<ov::op::v0::Parameter>(model_type, input_a_shape);
    auto input_b = std::make_shared<ov::op::v0::Parameter>(model_type, input_b_shape);

    auto transpose_order_a = ov::op::v0::Constant::create(ov::element::i64,
                                                          ov::Shape{target_transpose_order.size()},
                                                          target_transpose_order);
    auto transpose_a = std::make_shared<ov::op::v1::Transpose>(input_a, transpose_order_a);

    auto matmul = std::make_shared<ov::op::v0::MatMul>(transpose_a, input_b, true, true);

    auto model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input_a, input_b});
    return model;
}

void test_allowed_transpose_orders_with_matmul_and_inference(const std::vector<int64_t>& test_order) {
    ov::element::Type model_type = ov::element::f16;
    // 定义输入张量的形状和数据类型
    auto model = create_model(model_type, test_order);

    // 编译模型
    ov::Core core;
    auto compiled_model = core.compile_model(model, "GPU");

    // 创建推理请求
    auto infer_request = compiled_model.create_infer_request();

    // 动态调整输入数据的形状
    ov::Shape input_a_shape = {5, 6, 7, 8};  // 默认形状
    ov::Shape input_b_shape = {8, 7};        // 默认形状

    std::cout << "Input A shape: ";
    for (auto dim : input_a_shape) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;

    // 根据 test_order 调整 input_a_shape
    ov::Shape permuted_shape_a(input_a_shape.size());
    for (size_t i = 0; i < permuted_shape_a.size(); ++i) {
        permuted_shape_a[i] = input_a_shape[test_order[i]];
    }

    std::cout << "Input A shape after transposed: ";
    for (auto dim : permuted_shape_a) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;

    // 根据 test_order 调整 input_b_shape
    ov::Shape permuted_shape_b(input_b_shape.size());
    for (size_t i = 0; i < permuted_shape_b.size(); ++i) {
        permuted_shape_b[i] = input_a_shape[test_order[test_order.size() - 1 - i]];
    }
    // 这里假设 input_b_shape 的最后一个维度与 input_a_shape 的第一个维度相同
    std::cout << "Input B shape: ";
    for (auto dim : permuted_shape_b) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;

    // 准备输入数据
    auto input_a_data = generate_random_data(input_a_shape[0] * input_a_shape[1] * input_a_shape[2] *
                                             input_a_shape[3]);                           // 输入 A 的数据
    auto input_b_data = generate_random_data(permuted_shape_b[0] * permuted_shape_b[1]);  // 输入 B 的数据

    // 设置输入数据
    infer_request.set_input_tensor(0, ov::Tensor(model_type, input_a_shape, input_a_data.data()));
    infer_request.set_input_tensor(1, ov::Tensor(model_type, permuted_shape_b, input_b_data.data()));

    // 执行推理
    infer_request.infer();

    // 获取输出数据
    auto output_tensor = infer_request.get_output_tensor(0);
    auto output_data = output_tensor.data<float>();

    // 打印输出数据的前几个值（用于验证）
    std::cout << "Output data (first 5 values): ";
    for (size_t i = 0; i < std::min<size_t>(5, output_tensor.get_size()); ++i) {
        std::cout << output_data[i] << " ";
    }
    std::cout << std::endl;
}

void test_matmul_overflow_inference() {
    // 定义输入张量的形状和数据类型
    ov::element::Type model_type = ov::element::f32;
    auto model = create_2x2_matmul_model();

    // 编译模型
    ov::Core core;
    auto compiled_model = core.compile_model(model, "GPU", {{"ACTIVATIONS_SCALE_FACTOR", 8}, {"INFERENCE_PRECISION_HINT", "FP16"}});
    std::cout << "Model compiled successfully for GPU with FP16 precision." << std::endl;

    // 创建推理请求
    auto infer_request = compiled_model.create_infer_request();

    // 准备输入数据
    std::vector<float> input_a_data = {400.0f, 400.0f, 400.0f, 1.0f};  // 输入 A 的数据
    std::vector<float> input_b_data = {240.0f, 240.0f, 240.0f, 1.0f};  // 输入 B 的数据

    // 设置输入数据
    infer_request.set_input_tensor(0, ov::Tensor(model_type, ov::Shape{2, 2}, input_a_data.data()));
    infer_request.set_input_tensor(1, ov::Tensor(model_type, ov::Shape{2, 2}, input_b_data.data()));

    print_2x2_matrix(input_a_data.data(), "Input A");
    print_2x2_matrix(input_b_data.data(), "Input B");
    // 执行推理
    infer_request.infer();
    std::cout << "Inference completed successfully." << std::endl;

    // 获取输出数据
    auto output_tensor = infer_request.get_output_tensor(0);
    auto output_data = output_tensor.data<float>();

    // 打印输出数据的前几个值（用于验证）
    print_2x2_matrix(output_data, "Output data after matmul");
}

int main() {
    // 定义 allowed_orders 中的转置顺序
    //const std::vector<std::vector<int64_t>> allowed_orders = {
    //    {0, 3, 1, 2},

    //    {0, 1, 2, 3},
    //    {0, 1, 3, 2},
    //    {1, 2, 3, 0},
    //    {0, 2, 1, 3},


    //    {1, 2, 0, 3},
    //};
    //for (auto& order : allowed_orders) {
    //    std::cout << "===== Testing transpose order: ";
    //    for (auto dim : order) {
    //        std::cout << dim << " ";
    //    }
    //    std::cout << std::endl;

    //    try {
    //        test_allowed_transpose_orders_with_matmul_and_inference(order);
    //        std::cout << "PASS\n" << std::endl;
    //    } catch (const std::exception& e) {
    //        std::cerr << "Test failed: " << e.what() << std::endl;
    //    }
    //}
    test_matmul_overflow_inference();
    return 0;
}