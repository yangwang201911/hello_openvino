// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * CDPruner Implementation using OpenVINO Operations
 *
 * This file demonstrates how to implement CDPruner's core functionality
 * using OpenVINO's standard operations for maximum GPU acceleration.
 *
 * Compilation:
 * g++ -std=c++17 cdpruner_openvino_ops.cpp -lopenvino -o cdpruner_demo
 *
 * Or with CMake:
 * find_package(OpenVINO REQUIRED)
 * target_link_libraries(cdpruner_demo openvino::runtime)
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <openvino/op/ops.hpp>
#include <openvino/openvino.hpp>
#include <random>
#include <tuple>
#include <vector>

namespace ov::genai::cdpruner {

class CDPrunerOpsDemo {
private:
    ov::Core core;
    ov::CompiledModel unified_model;

public:
    // Initialize the models
    void initialize(const std::string& device = "CPU") {
        std::cout << "=== Initializing CDPruner with OpenVINO Ops ===" << std::endl;

        // Create and compile unified model
        auto unified_model_ptr = create_model();
        unified_model = core.compile_model(unified_model_ptr, device);
        std::cout << "CDPruner model compiled for device: " << device << std::endl;

        std::cout << "CDPruner initialization completed!" << std::endl;
    }

    // Main CDPruner processing function (optimized for performance testing)
    std::vector<std::vector<size_t>> process(const ov::Tensor& visual_features,
                                             const ov::Tensor& text_features,
                                             size_t num_tokens_to_select,
                                             bool verbose = false) {
        if (verbose) {
            std::cout << "\n=== CDPruner Processing ===" << std::endl;
        }

        // Single unified inference
        auto [relevance_scores, kernel_matrix, similarity_matrix] = compute(visual_features, text_features, verbose);
        if (verbose) {
            std::cout << "Step 1&2: relevance and kernel computation completed" << std::endl;
        }

        // Step 3: DPP selection (CPU implementation for complex logic)
        auto selected_tokens = dpp_select(kernel_matrix, num_tokens_to_select, verbose);
        if (verbose) {
            std::cout << "Step 3: DPP token selection completed" << std::endl;
        }

        return selected_tokens;
    }

    // Make compute public for testing
    std::tuple<ov::Tensor, ov::Tensor, ov::Tensor> compute(const ov::Tensor& visual_features,
                                                           const ov::Tensor& text_features,
                                                           bool verbose = false) {
        auto request = unified_model.create_infer_request();
        request.set_input_tensor(0, visual_features);
        request.set_input_tensor(1, text_features);
        request.infer();

        // Get all outputs for debugging
        auto relevance_scores = request.get_output_tensor(0);
        auto kernel_matrix = request.get_output_tensor(1);
        auto visual_self_similarity_matrix = request.get_output_tensor(2);

        if (verbose) {
            // Print intermediate results for debugging
            print_visual_similarity_matrix(visual_self_similarity_matrix);

            // Print relevance scores
            print_relevance_scores(relevance_scores);

            // Print kernel matrix
            print_kernel_matrix(kernel_matrix);
        }

        return {relevance_scores, kernel_matrix, visual_self_similarity_matrix};
    }

private:
    // Create unified model that computes both relevance and kernel in one pass
    std::shared_ptr<ov::Model> create_model() {
        // Input parameters
        auto visual_input =
            std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, -1});  // [B, N, D]
        auto text_input =
            std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1});  // [M, D]

        // ========== RELEVANCE COMPUTATION ==========
        // Step 1.1: L2 normalize visual features (will be reused for kernel computation)
        auto visual_l2_norm = create_l2_normalize(visual_input, -1);

        // Step 1.2: L2 normalize text features
        auto text_l2_norm = create_l2_normalize(text_input, -1);

        // Step 1.3: Compute similarity matrix [B, N, M]
        auto text_transposed = std::make_shared<ov::op::v1::Transpose>(
            text_l2_norm,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0}));

        auto text_visual_similarity = std::make_shared<ov::op::v0::MatMul>(visual_l2_norm, text_transposed);

        // Step 1.4: Compute mean similarity over text dimension
        auto mean_similarity = std::make_shared<ov::op::v1::ReduceMean>(
            text_visual_similarity,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}),  // axis=2
            false                                                               // keep_dims=false
        );

        // Step 1.5: Apply negative transformation only for CLIP-based models
        // Ref link: https://github.com/Theia-4869/CDPruner/issues/9
        auto negative_mean = std::make_shared<ov::op::v1::Multiply>(
            mean_similarity,
            ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {-1.0f}));

        // Step 1.6: Min-max normalization to get relevance scores
        auto relevance_scores = create_min_max_normalize(negative_mean);

        // Add debug outputs for intermediate results
        auto relevance_result = std::make_shared<ov::op::v0::Result>(relevance_scores);

        // ========== KERNEL COMPUTATION ==========
        // Step 2.1: Compute visual self-similarity matrix [B, N, N]
        auto visual_transposed = std::make_shared<ov::op::v1::Transpose>(
            visual_l2_norm,  // Reuse normalized visual features
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1}));

        auto visual_self_similarity = std::make_shared<ov::op::v0::MatMul>(visual_l2_norm, visual_transposed);

        auto visual_self_similarity_result = std::make_shared<ov::op::v0::Result>(visual_self_similarity);
        // Step 2.2: Build conditional kernel matrix
        auto relevance_i = std::make_shared<ov::op::v0::Unsqueeze>(
            relevance_scores,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2})  // axis=2
        );
        auto relevance_j = std::make_shared<ov::op::v0::Unsqueeze>(
            relevance_scores,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1})  // axis=1
        );

        // Conditional kernel: relevance[i] * similarity[i,j] * relevance[j]
        auto temp_kernel = std::make_shared<ov::op::v1::Multiply>(relevance_i, visual_self_similarity);
        auto conditional_kernel = std::make_shared<ov::op::v1::Multiply>(temp_kernel, relevance_j);

        // Create outputs for both relevance and kernel
        auto kernel_result = std::make_shared<ov::op::v0::Result>(conditional_kernel);

        return std::make_shared<ov::Model>(
            ov::ResultVector{relevance_result, kernel_result, visual_self_similarity_result},
            ov::ParameterVector{visual_input, text_input},
            "CDPruner_Model");
    }

    // Helper function: Create L2 normalization subgraph
    std::shared_ptr<ov::Node> create_l2_normalize(std::shared_ptr<ov::Node> input, int axis) {
        // Calculate squared values
        auto squared = std::make_shared<ov::op::v1::Multiply>(input, input);

        // Sum along the specified axis
        auto sum_squared = std::make_shared<ov::op::v1::ReduceSum>(
            squared,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {axis}),
            true  // keep_dims=true
        );

        // Add small epsilon for numerical stability
        auto epsilon = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1e-8f});
        auto sum_with_eps = std::make_shared<ov::op::v1::Add>(sum_squared, epsilon);

        // Square root to get L2 norm
        auto l2_norm = std::make_shared<ov::op::v0::Sqrt>(sum_with_eps);

        // Divide input by L2 norm
        return std::make_shared<ov::op::v1::Divide>(input, l2_norm);
    }

    // Helper function: Create min-max normalization subgraph
    std::shared_ptr<ov::Node> create_min_max_normalize(std::shared_ptr<ov::Node> input) {
        // Find min and max values along the last dimension
        auto min_vals = std::make_shared<ov::op::v1::ReduceMin>(
            input,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}),  // axis=1
            true                                                                // keep_dims=true
        );
        auto max_vals = std::make_shared<ov::op::v1::ReduceMax>(
            input,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}),  // axis=1
            true                                                                // keep_dims=true
        );

        // Compute range (max - min)
        auto range = std::make_shared<ov::op::v1::Subtract>(max_vals, min_vals);

        // Add small epsilon to avoid division by zero
        auto epsilon = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1e-8f});
        auto range_with_eps = std::make_shared<ov::op::v1::Add>(range, epsilon);

        // Normalize: (input - min) / (max - min)
        auto shifted = std::make_shared<ov::op::v1::Subtract>(input, min_vals);
        return std::make_shared<ov::op::v1::Divide>(shifted, range_with_eps);
    }

    // Helper function to print similarity information
    void print_visual_similarity_matrix(const ov::Tensor& visual_similarity_matrix) {
        // Print visual similarity matrix
        std::cout << "\n--- Visual similarity matrix ---" << std::endl;
        auto sim_shape = visual_similarity_matrix.get_shape();
        const float* sim_data = visual_similarity_matrix.data<const float>();
        std::cout << "Shape: [" << sim_shape[0] << ", " << sim_shape[1] << ", " << sim_shape[2] << "]" << std::endl;
        for (size_t i = 0; i < sim_shape[1]; ++i) {
            std::cout << "  Visual token " << i << ": [";
            for (size_t j = 0; j < sim_shape[2]; ++j) {
                if (j > 0)
                    std::cout << ", ";
                size_t idx = i * sim_shape[2] + j;
                std::cout << std::fixed << std::setprecision(4) << sim_data[idx];
            }
            std::cout << "]" << std::endl;
        }
    }

    // Helper function to print relevance scores
    void print_relevance_scores(const ov::Tensor& relevance_scores) {
        auto shape = relevance_scores.get_shape();
        const float* data = relevance_scores.data<const float>();

        std::cout << "\n--- Relevance Scores ---" << std::endl;
        std::cout << "Shape: [";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0)
                std::cout << ", ";
            std::cout << shape[i];
        }
        std::cout << "]" << std::endl;

        size_t batch_size = shape[0];
        size_t num_tokens = shape[1];

        for (size_t b = 0; b < batch_size; ++b) {
            std::cout << "Batch " << b << " relevance scores: [";
            for (size_t i = 0; i < num_tokens; ++i) {
                if (i > 0)
                    std::cout << ", ";
                size_t idx = b * num_tokens + i;
                std::cout << std::fixed << std::setprecision(4) << data[idx];
            }
            std::cout << "]" << std::endl;
        }
    }

    // Helper function to print kernel matrix
    void print_kernel_matrix(const ov::Tensor& kernel_matrix) {
        auto shape = kernel_matrix.get_shape();
        const float* data = kernel_matrix.data<const float>();

        std::cout << "\n--- Kernel Matrix ---" << std::endl;
        std::cout << "Shape: [";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0)
                std::cout << ", ";
            std::cout << shape[i];
        }
        std::cout << "]" << std::endl;

        size_t batch_size = shape[0];
        size_t num_tokens = shape[1];

        for (size_t b = 0; b < batch_size; ++b) {
            std::cout << "Batch " << b << " kernel matrix:" << std::endl;
            for (size_t i = 0; i < num_tokens; ++i) {
                std::cout << "  [";
                for (size_t j = 0; j < num_tokens; ++j) {
                    if (j > 0)
                        std::cout << ", ";
                    size_t idx = b * num_tokens * num_tokens + i * num_tokens + j;
                    std::cout << std::fixed << std::setprecision(4) << data[idx];
                }
                std::cout << "]" << std::endl;
            }
        }

        // Print diagonal elements (most important for DPP)
        std::cout << "Diagonal elements (token gains): [";
        for (size_t i = 0; i < num_tokens; ++i) {
            if (i > 0)
                std::cout << ", ";
            size_t diag_idx = i * num_tokens + i;
            std::cout << std::fixed << std::setprecision(4) << data[diag_idx];
        }
        std::cout << "]" << std::endl;
    }

    // Fast Greedy DPP selection (based on fast_dpp.cpp implementation)
    // Implements the efficient DPP greedy algorithm:
    // 1. Initialize marginal gains (diagonal elements)
    // 2. Select token with max gain
    // 3. Update orthogonal vectors and marginal gains efficiently
    // 4. Repeat until k tokens selected
    std::vector<std::vector<size_t>> dpp_select(const ov::Tensor& kernel_matrix,
                                                size_t num_tokens,
                                                bool verbose = false) {
        auto shape = kernel_matrix.get_shape();
        size_t batch_size = shape[0];
        size_t total_tokens = shape[1];

        if (verbose) {
            std::cout << "\n--- Fast Greedy DPP Selection Process ---" << std::endl;
            std::cout << "Kernel matrix shape: [" << batch_size << ", " << total_tokens << ", " << total_tokens << "]"
                      << std::endl;
            std::cout << "Tokens to select: " << num_tokens << std::endl;
        }

        const float* kernel_data = kernel_matrix.data<const float>();
        std::vector<std::vector<size_t>> results(batch_size);

        for (size_t b = 0; b < batch_size; ++b) {
            results[b] = select_single_batch(kernel_matrix, b, num_tokens, verbose);
        }

        return results;
    }

private:
    // Single batch DPP selection (following fast_dpp.cpp logic)
    std::vector<size_t> select_single_batch(const ov::Tensor& kernel,
                                            size_t batch_idx,
                                            size_t num_tokens,
                                            bool verbose = false) {
        auto shape = kernel.get_shape();
        size_t total_tokens = shape[1];

        if (verbose) {
            std::cout << "\nBatch " << batch_idx << " Fast DPP selection:" << std::endl;
        }

        // Initialize marginal gains (diagonal elements) [total_tokens]
        std::vector<float> di2s(total_tokens);

        // Copy diagonal elements from kernel for this batch
        const float* kernel_data = kernel.data<const float>();
        for (size_t i = 0; i < total_tokens; ++i) {
            size_t diag_idx = batch_idx * total_tokens * total_tokens + i * total_tokens + i;
            di2s[i] = kernel_data[diag_idx];
        }

        if (verbose) {
            std::cout << "Initial marginal gains (diagonal): [";
            for (size_t i = 0; i < total_tokens; ++i) {
                if (i > 0)
                    std::cout << ", ";
                std::cout << std::fixed << std::setprecision(4) << di2s[i];
            }
            std::cout << "]" << std::endl;
        }

        // Orthogonalized vectors [num_tokens, total_tokens]
        // cis[t][j] = orthogonalized vector for t-th selected token
        std::vector<std::vector<float>> cis(num_tokens, std::vector<float>(total_tokens, 0.0f));

        std::vector<size_t> selected_indices;
        selected_indices.reserve(num_tokens);

        // Greedy selection loop - core Fast DPP algorithm
        for (size_t t = 0; t < num_tokens; ++t) {
            if (verbose) {
                std::cout << "\nSelection round " << (t + 1) << ":" << std::endl;
                std::cout << "  Current marginal gains: [";
                for (size_t i = 0; i < total_tokens; ++i) {
                    if (i > 0)
                        std::cout << ", ";
                    if (std::find(selected_indices.begin(), selected_indices.end(), i) != selected_indices.end()) {
                        std::cout << "used";
                    } else {
                        std::cout << std::fixed << std::setprecision(4) << di2s[i];
                    }
                }
                std::cout << "]" << std::endl;
            }

            // Find the token with maximum marginal gain
            size_t best_idx = argmax(di2s, selected_indices);
            selected_indices.push_back(best_idx);

            if (verbose) {
                std::cout << "  Selected token " << best_idx << " with gain " << std::fixed << std::setprecision(4)
                          << di2s[best_idx] << std::endl;
            }

            // Update orthogonal vector and marginal gains
            if (t < num_tokens - 1) {  // Don't update after last selection
                update_orthogonal_vector(kernel, batch_idx, best_idx, t, cis, di2s, verbose);
                update_marginal_gains(t, best_idx, cis, di2s, verbose);

                // Set selected token's gain to negative infinity to prevent re-selection
                di2s[best_idx] = -std::numeric_limits<float>::infinity();
            }
        }

        // Sort the selected indices for deterministic output
        std::sort(selected_indices.begin(), selected_indices.end());

        if (verbose) {
            std::cout << "Final selected tokens (sorted): [";
            for (size_t i = 0; i < selected_indices.size(); ++i) {
                if (i > 0)
                    std::cout << ", ";
                std::cout << selected_indices[i];
            }
            std::cout << "]" << std::endl;
        }

        return selected_indices;
    }

    // Find argmax excluding already selected indices
    size_t argmax(const std::vector<float>& scores, const std::vector<size_t>& selected_indices) {
        size_t best_idx = 0;
        float best_value = -std::numeric_limits<float>::infinity();

        for (size_t i = 0; i < scores.size(); ++i) {
            // Skip already selected tokens
            if (std::find(selected_indices.begin(), selected_indices.end(), i) != selected_indices.end()) {
                continue;
            }

            if (scores[i] > best_value) {
                best_value = scores[i];
                best_idx = i;
            }
        }

        return best_idx;
    }

    // Update orthogonal vector (following fast_dpp.cpp logic)
    void update_orthogonal_vector(const ov::Tensor& kernel,
                                  size_t batch_idx,
                                  size_t selected_idx,
                                  size_t iteration,
                                  std::vector<std::vector<float>>& cis,
                                  const std::vector<float>& di2s,
                                  bool verbose = false) {
        // eis = (kernel[batch, selected_idx] - sum(cis[:iteration] * cis[:iteration, selected_idx])) /
        // sqrt(di2s[selected_idx])

        auto kernel_shape = kernel.get_shape();
        size_t total_tokens = kernel_shape[1];

        const float* kernel_data = kernel.data<const float>();

        // Get normalization factor
        float norm_factor = std::sqrt(std::max(di2s[selected_idx], 1e-8f));

        if (verbose) {
            std::cout << "  Orthogonalization step:" << std::endl;
            std::cout << "    Selected token " << selected_idx << " with gain " << std::fixed << std::setprecision(4)
                      << di2s[selected_idx] << ", norm_factor = " << std::fixed << std::setprecision(4) << norm_factor
                      << std::endl;
        }

        // Compute the new orthogonal vector for each token
        for (size_t j = 0; j < total_tokens; ++j) {
            // Get kernel[batch_idx, selected_idx, j]
            size_t kernel_idx = batch_idx * total_tokens * total_tokens + selected_idx * total_tokens + j;
            float kernel_val = kernel_data[kernel_idx];

            // Subtract projection onto previously selected vectors
            // sum(cis[:iteration, selected_idx] * cis[:iteration, j])
            float projection = 0.0f;
            for (size_t prev_t = 0; prev_t < iteration; ++prev_t) {
                projection += cis[prev_t][selected_idx] * cis[prev_t][j];
            }

            // Store the orthogonalized vector element
            cis[iteration][j] = (kernel_val - projection) / norm_factor;
        }

        if (verbose) {
            std::cout << "    New orthogonal vector cis[" << iteration << "]: [";
            for (size_t j = 0; j < std::min(total_tokens, size_t(8)); ++j) {  // Show first 8 elements
                if (j > 0)
                    std::cout << ", ";
                std::cout << std::fixed << std::setprecision(4) << cis[iteration][j];
            }
            if (total_tokens > 8)
                std::cout << ", ...";
            std::cout << "]" << std::endl;
        }
    }

    // Update marginal gains (following fast_dpp.cpp logic)
    void update_marginal_gains(size_t iteration,
                               size_t selected_idx,
                               const std::vector<std::vector<float>>& cis,
                               std::vector<float>& di2s,
                               bool verbose = false) {
        // di2s -= square(eis) where eis is the newly computed orthogonal vector cis[iteration]

        size_t total_tokens = cis[iteration].size();

        if (verbose) {
            std::cout << "    Marginal gains update:" << std::endl;
            std::cout << "      Before update: [";
            for (size_t j = 0; j < total_tokens; ++j) {
                if (j > 0)
                    std::cout << ", ";
                std::cout << std::fixed << std::setprecision(4) << di2s[j];
            }
            std::cout << "]" << std::endl;
        }

        // Update marginal gains for all tokens
        for (size_t j = 0; j < total_tokens; ++j) {
            float eis_j = cis[iteration][j];
            float old_gain = di2s[j];

            // Subtract the squared orthogonal component
            di2s[j] -= eis_j * eis_j;

            if (verbose && j < 8) {  // Show details for first 8 tokens
                std::cout << "      Token " << j << ": " << std::fixed << std::setprecision(4) << old_gain << " - "
                          << std::fixed << std::setprecision(4) << (eis_j * eis_j) << " = " << std::fixed
                          << std::setprecision(4) << di2s[j] << std::endl;
            }
        }

        if (verbose) {
            std::cout << "      After update: [";
            for (size_t j = 0; j < total_tokens; ++j) {
                if (j > 0)
                    std::cout << ", ";
                std::cout << std::fixed << std::setprecision(4) << di2s[j];
            }
            std::cout << "]" << std::endl;
        }
    }
};

// Helper function to create test data
std::pair<ov::Tensor, ov::Tensor> create_test_data() {
    std::cout << "\n=== Creating Test Data ===" << std::endl;

    // Parameters
    const size_t batch_size = 1;
    const size_t num_visual_tokens = 4;
    const size_t visual_feature_dim = 3;
    const size_t num_text_tokens = 2;
    const size_t text_feature_dim = 3;

    // Create visual features [1, 4, 3]
    ov::Tensor visual_features(ov::element::f32, {batch_size, num_visual_tokens, visual_feature_dim});
    float* visual_data = visual_features.data<float>();

    // Sample visual features (representing 4 visual tokens with 3D features)
    std::vector<std::vector<float>> visual_values = {
        {1.0f, 2.0f, 1.0f},  // Token 0: moderate relevance
        {3.0f, 1.0f, 2.0f},  // Token 1: high relevance
        {0.5f, 0.5f, 3.0f},  // Token 2: low relevance
        {2.0f, 2.0f, 1.5f}   // Token 3: moderate relevance
    };

    for (size_t i = 0; i < num_visual_tokens; ++i) {
        for (size_t j = 0; j < visual_feature_dim; ++j) {
            visual_data[i * visual_feature_dim + j] = visual_values[i][j];
        }
    }

    // Create text features [2, 3]
    ov::Tensor text_features(ov::element::f32, {num_text_tokens, text_feature_dim});
    float* text_data = text_features.data<float>();

    // Sample text features (representing 2 text tokens)
    std::vector<std::vector<float>> text_values = {
        {2.5f, 1.5f, 1.8f},  // Text token 0
        {1.2f, 2.8f, 2.0f}   // Text token 1
    };

    for (size_t i = 0; i < num_text_tokens; ++i) {
        for (size_t j = 0; j < text_feature_dim; ++j) {
            text_data[i * text_feature_dim + j] = text_values[i][j];
        }
    }

    std::cout << "Visual features shape: [" << batch_size << ", " << num_visual_tokens << ", " << visual_feature_dim
              << "]" << std::endl;
    std::cout << "Text features shape: [" << num_text_tokens << ", " << text_feature_dim << "]" << std::endl;

    // Print visual features
    std::cout << "\nVisual features:" << std::endl;
    for (size_t i = 0; i < num_visual_tokens; ++i) {
        std::cout << "  Token " << i << ": [";
        for (size_t j = 0; j < visual_feature_dim; ++j) {
            if (j > 0)
                std::cout << ", ";
            std::cout << std::fixed << std::setprecision(2) << visual_values[i][j];
        }
        std::cout << "]" << std::endl;
    }

    // Print text features
    std::cout << "\nText features:" << std::endl;
    for (size_t i = 0; i < num_text_tokens; ++i) {
        std::cout << "  Token " << i << ": [";
        for (size_t j = 0; j < text_feature_dim; ++j) {
            if (j > 0)
                std::cout << ", ";
            std::cout << std::fixed << std::setprecision(2) << text_values[i][j];
        }
        std::cout << "]" << std::endl;
    }

    return {visual_features, text_features};
}

// Helper function to print results
void print_results(const std::vector<std::vector<size_t>>& selected_tokens, size_t num_tokens_to_select) {
    std::cout << "\n=== CDPruner Results ===" << std::endl;
    std::cout << "Requested tokens to select: " << num_tokens_to_select << std::endl;

    for (size_t b = 0; b < selected_tokens.size(); ++b) {
        std::cout << "Batch " << b << " selected tokens: [";
        for (size_t i = 0; i < selected_tokens[b].size(); ++i) {
            if (i > 0)
                std::cout << ", ";
            std::cout << selected_tokens[b][i];
        }
        std::cout << "]" << std::endl;
    }
}

}  // namespace ov::genai::cdpruner

// Test case class with expected outputs
class CDPrunerAccuracyTest {
private:
    ov::genai::cdpruner::CDPrunerOpsDemo pruner;

    // Expected outputs for the standard test data
    std::vector<float> expected_relevance_scores = {0.1248f, 0.3406f, 1.0000f, 0.0000f};

    // Expected visual self-similarity matrix (using L2 normalized features)
    std::vector<std::vector<float>> expected_visual_similarity = {
        {1.000f, 0.763f, 0.595f, 0.956f},  // Token 0 similarities
        {0.763f, 1.000f, 0.694f, 0.919f},  // Token 1 similarities
        {0.595f, 0.694f, 1.000f, 0.658f},  // Token 2 similarities
        {0.956f, 0.919f, 0.658f, 1.000f}   // Token 3 similarities
    };

    // Expected kernel matrix: relevance[i] * similarity[i,j] * relevance[j]
    std::vector<std::vector<float>> expected_kernel_matrix = {
        {0.0156f, 0.0324f, 0.0743f, 0.0000f},  // Row 0
        {0.0324f, 0.1160f, 0.2364f, 0.0000f},  // Row 1
        {0.0743f, 0.2364f, 1.0000f, 0.0000f},  // Row 2
        {0.0000f, 0.0000f, 0.0000f, 0.0000f}   // Row 3
    };

public:
    void initialize() {
        std::cout << "\n=== CDPruner Accuracy Test ===" << std::endl;
        pruner.initialize("GPU");
    }

    void run_accuracy_test() {
        std::cout << "\n--- Testing Expected Outputs ---" << std::endl;

        // Use the standard test data
        auto [visual_features, text_features] = ov::genai::cdpruner::create_test_data();

        // Run the model
        auto [relevance_scores, kernel_matrix, visual_self_similarity_matrix] =
            pruner.compute(visual_features, text_features, true);

        // Test visual self-similarity matrix
        test_visual_similarity_matrix(visual_self_similarity_matrix);

        // Test relevance scores
        test_relevance_scores(relevance_scores);

        // Test kernel matrix
        test_kernel_matrix(kernel_matrix);

        // Test full pipeline
        test_full_pipeline(visual_features, text_features);
    }

private:
    void test_visual_similarity_matrix(const ov::Tensor& actual_visual_similarity) {
        std::cout << "\n--- Testing Visual Self-Similarity Matrix ---" << std::endl;

        const float* actual_data = actual_visual_similarity.data<const float>();
        auto shape = actual_visual_similarity.get_shape();
        float tolerance = 0.01f;  // 1% tolerance for floating point precision

        std::cout << "Expected visual self-similarity matrix [4x4]:" << std::endl;
        for (size_t i = 0; i < 4; ++i) {
            std::cout << "  Visual token " << i << ": [";
            for (size_t j = 0; j < 4; ++j) {
                if (j > 0)
                    std::cout << ", ";
                std::cout << std::fixed << std::setprecision(4) << expected_visual_similarity[i][j];
            }
            std::cout << "]" << std::endl;
        }

        std::cout << "\nActual visual self-similarity matrix:" << std::endl;
        for (size_t i = 0; i < shape[1]; ++i) {  // visual tokens (rows)
            std::cout << "  Visual token " << i << ": [";
            for (size_t j = 0; j < shape[2]; ++j) {  // visual tokens (cols)
                if (j > 0)
                    std::cout << ", ";
                size_t idx = i * shape[2] + j;
                std::cout << std::fixed << std::setprecision(4) << actual_data[idx];
            }
            std::cout << "]" << std::endl;
        }

        // Test each element
        bool all_passed = true;
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                size_t idx = i * 4 + j;
                float diff = std::abs(actual_data[idx] - expected_visual_similarity[i][j]);
                if (diff > tolerance) {
                    std::cout << "✗ Visual similarity[" << i << "," << j
                              << "]: FAIL (expected: " << expected_visual_similarity[i][j]
                              << ", actual: " << actual_data[idx] << ", diff: " << diff << ")" << std::endl;
                    all_passed = false;
                }
            }
        }

        if (all_passed) {
            std::cout << "✓ Visual self-similarity matrix matches expected values!" << std::endl;
        } else {
            std::cout << "⚠️ Some visual self-similarity values don't match expected values." << std::endl;
        }
    }

    void test_relevance_scores(const ov::Tensor& actual_relevance) {
        std::cout << "\n--- Testing Relevance Scores ---" << std::endl;

        const float* actual_data = actual_relevance.data<const float>();
        float tolerance = 0.01f;  // 1% tolerance for floating point precision

        std::cout << "Expected: [";
        for (size_t i = 0; i < expected_relevance_scores.size(); ++i) {
            if (i > 0)
                std::cout << ", ";
            std::cout << std::fixed << std::setprecision(4) << expected_relevance_scores[i];
        }
        std::cout << "]" << std::endl;

        std::cout << "Actual:   [";
        for (size_t i = 0; i < 4; ++i) {
            if (i > 0)
                std::cout << ", ";
            std::cout << std::fixed << std::setprecision(4) << actual_data[i];
        }
        std::cout << "]" << std::endl;

        bool all_passed = true;
        for (size_t i = 0; i < 4; ++i) {
            float diff = std::abs(actual_data[i] - expected_relevance_scores[i]);
            if (diff <= tolerance) {
                std::cout << "✓ Token " << i << " relevance: PASS (diff: " << diff << ")" << std::endl;
            } else {
                std::cout << "✗ Token " << i << " relevance: FAIL (diff: " << diff << ", tolerance: " << tolerance
                          << ")" << std::endl;
                all_passed = false;
            }
        }

        if (all_passed) {
            std::cout << "✓ All relevance scores match expected values!" << std::endl;
        } else {
            std::cout << "✗ Some relevance scores don't match expected values." << std::endl;
        }
    }

    void test_kernel_matrix(const ov::Tensor& actual_kernel) {
        std::cout << "\n--- Testing Kernel Matrix ---" << std::endl;

        const float* actual_data = actual_kernel.data<const float>();
        float tolerance = 0.01f;

        std::cout << "Expected kernel matrix:" << std::endl;
        for (size_t i = 0; i < 4; ++i) {
            std::cout << "  [";
            for (size_t j = 0; j < 4; ++j) {
                if (j > 0)
                    std::cout << ", ";
                std::cout << std::fixed << std::setprecision(4) << expected_kernel_matrix[i][j];
            }
            std::cout << "]" << std::endl;
        }

        std::cout << "Actual kernel matrix:" << std::endl;
        for (size_t i = 0; i < 4; ++i) {
            std::cout << "  [";
            for (size_t j = 0; j < 4; ++j) {
                if (j > 0)
                    std::cout << ", ";
                size_t idx = i * 4 + j;
                std::cout << std::fixed << std::setprecision(4) << actual_data[idx];
            }
            std::cout << "]" << std::endl;
        }

        bool all_passed = true;
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                size_t idx = i * 4 + j;
                float diff = std::abs(actual_data[idx] - expected_kernel_matrix[i][j]);
                if (diff > tolerance) {
                    std::cout << "✗ Kernel[" << i << "," << j << "]: FAIL (expected: " << expected_kernel_matrix[i][j]
                              << ", actual: " << actual_data[idx] << ", diff: " << diff << ")" << std::endl;
                    all_passed = false;
                }
            }
        }

        if (all_passed) {
            std::cout << "✓ Kernel matrix matches expected values!" << std::endl;
        } else {
            std::cout << "⚠️ Some kernel matrix values don't match expected values." << std::endl;
        }
    }

    void test_full_pipeline(const ov::Tensor& visual_features, const ov::Tensor& text_features) {
        std::cout << "\n--- Testing Full Pipeline ---" << std::endl;

        // Test token selection
        auto selected_2 = pruner.process(visual_features, text_features, 2, false);
        auto selected_3 = pruner.process(visual_features, text_features, 3, false);

        std::cout << "Selected 2 tokens: [";
        for (size_t i = 0; i < selected_2[0].size(); ++i) {
            if (i > 0)
                std::cout << ", ";
            std::cout << selected_2[0][i];
        }
        std::cout << "]" << std::endl;

        std::cout << "Selected 3 tokens: [";
        for (size_t i = 0; i < selected_3[0].size(); ++i) {
            if (i > 0)
                std::cout << ", ";
            std::cout << selected_3[0][i];
        }
        std::cout << "]" << std::endl;

        // Based on relevance scores [0.1248, 0.3406, 1.0000, 0.0000]
        // Token 2 should always be selected (highest relevance)
        // Token 1 should be selected next (second highest)
        bool token_2_selected = std::find(selected_2[0].begin(), selected_2[0].end(), 2) != selected_2[0].end();
        bool token_1_selected = std::find(selected_2[0].begin(), selected_2[0].end(), 1) != selected_2[0].end();

        if (token_2_selected) {
            std::cout << "✓ Highest relevance token (2) is selected" << std::endl;
        } else {
            std::cout << "✗ Highest relevance token (2) is NOT selected" << std::endl;
        }

        if (token_1_selected) {
            std::cout << "✓ Second highest relevance token (1) is selected" << std::endl;
        } else {
            std::cout << "✓ Token selection considers diversity (token 1 not selected due to DPP)" << std::endl;
        }
    }
};

// Performance benchmark function
void benchmark_device(const std::string& device,
                      const ov::Tensor& visual_features,
                      const ov::Tensor& text_features,
                      size_t num_iterations = 100) {
    std::cout << "\n=== Benchmarking " << device << " ===" << std::endl;

    try {
        // Initialize CDPruner for this device
        ov::genai::cdpruner::CDPrunerOpsDemo pruner;
        auto start_init = std::chrono::high_resolution_clock::now();
        pruner.initialize(device);
        auto end_init = std::chrono::high_resolution_clock::now();

        auto init_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_init - start_init);
        std::cout << "Initialization time: " << init_time.count() << " ms" << std::endl;

        // Warmup runs
        std::cout << "Performing warmup runs..." << std::endl;
        for (size_t i = 0; i < 5; ++i) {
            pruner.process(visual_features, text_features, 2);
        }

        // Actual benchmark
        std::cout << "Running " << num_iterations << " benchmark iterations..." << std::endl;
        std::vector<double> inference_times;

        for (size_t i = 0; i < num_iterations; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            auto selected_tokens = pruner.process(visual_features, text_features, 2);
            auto end = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            inference_times.push_back(duration.count() / 1000.0);  // Convert to milliseconds
        }

        // Calculate statistics
        double total_time = 0.0;
        double min_time = inference_times[0];
        double max_time = inference_times[0];

        for (double time : inference_times) {
            total_time += time;
            min_time = std::min(min_time, time);
            max_time = std::max(max_time, time);
        }

        double avg_time = total_time / num_iterations;

        // Calculate standard deviation
        double variance = 0.0;
        for (double time : inference_times) {
            variance += (time - avg_time) * (time - avg_time);
        }
        double std_dev = std::sqrt(variance / num_iterations);

        // Print results
        std::cout << "\n--- " << device << " Performance Results ---" << std::endl;
        std::cout << "Average inference time: " << std::fixed << std::setprecision(3) << avg_time << " ms" << std::endl;
        std::cout << "Min inference time: " << std::fixed << std::setprecision(3) << min_time << " ms" << std::endl;
        std::cout << "Max inference time: " << std::fixed << std::setprecision(3) << max_time << " ms" << std::endl;
        std::cout << "Standard deviation: " << std::fixed << std::setprecision(3) << std_dev << " ms" << std::endl;
        std::cout << "Throughput: " << std::fixed << std::setprecision(2) << (1000.0 / avg_time) << " inferences/second"
                  << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error benchmarking " << device << ": " << e.what() << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "====================================================" << std::endl;

    // Parse command line arguments
    std::string mode = "accuracy";  // default mode
    if (argc > 1) {
        mode = argv[1];
    }

    try {
        if (mode == "accuracy" || mode == "all") {
            // Run accuracy tests with expected outputs
            CDPrunerAccuracyTest accuracy_test;
            accuracy_test.initialize();
            accuracy_test.run_accuracy_test();
        }

        if (mode == "benchmark" || mode == "all") {
            // Run performance benchmarks
            std::cout << "\n=== Performance Benchmarks ===" << std::endl;
            auto [visual_features, text_features] = ov::genai::cdpruner::create_test_data();

            // Benchmark CPU
            benchmark_device("CPU", visual_features, text_features, 50);

            // Benchmark GPU (if available)
            benchmark_device("GPU", visual_features, text_features, 50);
        }

        if (mode == "demo") {
            // Run single demo with detailed output
            std::cout << "\n=== Demo Mode ===" << std::endl;
            ov::genai::cdpruner::CDPrunerOpsDemo pruner;
            pruner.initialize("GPU");

            auto [visual_features, text_features] = ov::genai::cdpruner::create_test_data();
            auto selected_tokens = pruner.process(visual_features, text_features, 2, true);  // verbose=true
            ov::genai::cdpruner::print_results(selected_tokens, 2);
        }

        if (mode != "accuracy" && mode != "benchmark" && mode != "all" && mode != "demo") {
            std::cout << "\nUsage: " << argv[0] << " [mode]" << std::endl;
            std::cout << "Modes:" << std::endl;
            std::cout << "  accuracy  - Run accuracy tests with expected outputs (default)" << std::endl;
            std::cout << "  benchmark - Run performance benchmarks" << std::endl;
            std::cout << "  demo      - Run demo with detailed output" << std::endl;
            std::cout << "  all       - Run both accuracy tests and benchmarks" << std::endl;
        }

        std::cout << "\n=== Completed Successfully! ===" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}