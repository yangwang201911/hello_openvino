#include <chrono>
#include <iostream>
#include <openvino/op/ops.hpp>
#include <openvino/openvino.hpp>

#include "create_model_funcs_list.hpp"
using namespace std;
int main(int argc, char* argv[]) {
    auto model = create_simple_model();
    std::string device_name;
    if (argc == 1)
        device_name = "CPU";
    else
        device_name = argv[1];

    ov::Core core;
    std::cout << ov::get_openvino_version() << std::endl;
    auto supported_properties = core.get_property(device_name, ov::supported_properties);
    std::string device_luid = core.get_property(device_name, ov::device::luid.name()).as<std::string>();
    std::cout << "Device LUID: " << device_luid << std::endl;
    std::cout << "====== Device properties: " << device_name << "======" << std::endl;
    for (const auto& cfg : supported_properties) {
        if (cfg == ov::supported_properties)
            continue;
        std::cout << " property: " << cfg << std::endl; 
        continue;
        auto prop = core.get_property(device_name, cfg);
        if (cfg == ov::device::properties) {
            auto devices_properties = prop.as<ov::AnyMap>();
            for (auto& item : devices_properties) {
                std::cout << "  " << item.first << ": " << std::endl;
                for (auto& item2 : item.second.as<ov::AnyMap>()) {
                    std::cout << "    " << item2.first << ": " << item2.second.as<std::string>() << std::endl;
                }
            }
        } else {
            std::cout << "  " << cfg << ": " << prop.as<std::string>() << std::endl;
        }
    }
    //std::cout << "======Will compiling model on device: ======" << device_name << std::endl;
    //{
    //    ov::Core core;
    //    std::cout << ov::get_openvino_version() << std::endl;
    //    auto compiled_model = core.compile_model(
    //        model,
    //        device_name,
    //        ov::device::priorities("GPU.1"));
    //}
    std::cout << "Compiled done.\n";

#if 0
    std::shared_ptr<ov::Model> model = core.read_model(argv[1]);
    std::string device = argv[2];
    if (device.find("AUTO:") != std::string::npos)
        core.set_property("AUTO", {ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY)});
    else
        core.set_property(device,
                          {ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY),
                           ov::hint::inference_precision(ov::element::undefined)});
    {
        ov::CompiledModel compiled_model;
        if (device.find("AUTO:") != std::string::npos)
            compiled_model = core.compile_model(model, device, {ov::hint::inference_precision(ov::element::undefined)});
        else
            compiled_model = core.compile_model(model, device);
        auto supported_properties = compiled_model.get_property(ov::supported_properties);
        std::cout << "Device: " << device << std::endl;
        for (const auto& cfg : supported_properties) {
            if (cfg == ov::supported_properties)
                continue;
            auto prop = compiled_model.get_property(cfg);
            if (cfg == ov::device::properties) {
                auto devices_properties = prop.as<ov::AnyMap>();
                for (auto& item : devices_properties) {
                    std::cout << "  " << item.first << ": " << std::endl;
                    for (auto& item2 : item.second.as<ov::AnyMap>()) {
                        std::cout << "    " << item2.first << ": " << item2.second.as<std::string>() << std::endl;
                    }
                }
            } else {
                std::cout << "  " << cfg << ": " << prop.as<std::string>() << std::endl;
            }
        }
    }
#endif
    return 0;
}