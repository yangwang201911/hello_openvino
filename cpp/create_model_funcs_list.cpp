#include "create_model_funcs_list.hpp"

#include <chrono>
#include <iostream>
#include <openvino/op/ops.hpp>
#include <openvino/openvino.hpp>
using namespace std;
std::shared_ptr<ov::Model> create_model_tensor_name_changed(const std::string save_path) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 77});
    param->set_friendly_name("input_ids");

    auto reshape_val = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 77});
    reshape_val->set_friendly_name("reshape_val");

    auto reshape = std::make_shared<ov::op::v1::Reshape>(param, reshape_val, true);
    reshape->set_friendly_name("reshape");

    auto convert = std::make_shared<ov::op::v0::Convert>(reshape, ov::element::i32);
    convert->set_friendly_name("convert");

    auto result = std::make_shared<ov::op::v0::Result>(convert);
    result->set_friendly_name("res");
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    if (!save_path.empty())
        ov::serialize(model, save_path);
    return model;
}

std::shared_ptr<ov::Model> create_simple_model(const std::string save_path) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 2, 2});
    param->set_friendly_name("input");
    param->get_output_tensor(0).set_names({"input_tensor"});
    auto const_value = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 1}, {1});
    const_value->set_friendly_name("const_val");
    auto add = std::make_shared<ov::op::v1::Add>(param, const_value);
    add->set_friendly_name("add");
    auto result = std::make_shared<ov::op::v0::Result>(add);
    result->set_friendly_name("res");
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    if (!save_path.empty())
        ov::serialize(model, save_path);
    return model;
}

std::shared_ptr<ov::Model> create_dynamic_model(const std::string save_path) {
    ov::PartialShape shape({ov::Dimension::dynamic(), 2});
    ov::element::Type type(ov::element::Type_t::f32);
    auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
    param->get_output_tensor(0).set_names({"tensor"});
    auto relu = std::make_shared<ov::op::v0::Relu>(param);
    auto result = std::make_shared<ov::op::v0::Result>(relu);

    ov::ParameterVector params = {param};
    ov::ResultVector results = {result};

    auto model = std::make_shared<ov::Model>(results, params);
    if (!save_path.empty())
        ov::serialize(model, save_path);
    return model;
}

std::shared_ptr<ov::Model> create_dynamic_output_model() {
    auto boxes = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 4});
    boxes->set_friendly_name("param_1");
    boxes->get_output_tensor(0).set_names({"input_tensor_1"});

    auto scores = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 2}, {1});
    scores->set_friendly_name("const_val");


    // auto scores = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 2});
    // scores->set_friendly_name("param_2");
    // scores->get_output_tensor(0).set_names({"input_tensor_2"});
    auto max_output_boxes_per_class = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {10});
    auto iou_threshold = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.5});
    auto score_threshold = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.5});
    auto nms = std::make_shared<ov::op::v9::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold);
    auto res = std::make_shared<ov::op::v0::Result>(nms);
    res->set_friendly_name("output_dynamic");
    return std::make_shared<ov::Model>(ov::NodeVector{nms}, ov::ParameterVector{boxes});
}