import os
import sys
import time
import numpy as np
from transformers import AutoTokenizer

# 原生 OpenVINO API
import openvino as ov

class Model_Qwen3_1_7B_Native_OV():
    def __init__(self, model_path, device="GPU.1"):
        print(f"== Loading model from path: {model_path}")
        print(f"== Using device: {device}")
        
        # 初始化 OpenVINO Core
        self.core = ov.Core()
        
        # 加载模型
        model_xml = os.path.join(model_path, "openvino_model.xml")
        model_bin = os.path.join(model_path, "openvino_model.bin")
        
        # 读取模型
        self.model = self.core.read_model(model_xml)
        
        # 编译模型到指定设备
        self.compiled_model = self.core.compile_model(self.model, device)
        
        # 获取输入输出信息
        self.inputs = self.compiled_model.inputs
        self.outputs = self.compiled_model.outputs
        
        # 打印所有输入信息
        print("=== Model Inputs ===")
        for i, input_layer in enumerate(self.inputs):
            print(f"Input {i}: name={input_layer.any_name}, partial_shape={input_layer.partial_shape}, type={input_layer.element_type}")
        
        print("=== Model Outputs ===")
        for i, output_layer in enumerate(self.outputs):
            print(f"Output {i}: name={output_layer.any_name}, partial_shape={output_layer.partial_shape}, type={output_layer.element_type}")
        
        # 加载 tokenizer（使用标准 transformers）
        self.__tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    def infer(self, desc, max_new_tokens=1024):
        target_lang = "英文"
        prompt = f"请把下面的句子从{desc}翻译成{target_lang}，请直接输出{target_lang}："
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = self.__tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False
        )
        print(f"Initial prompt: {text}")

        # 初始tokenize
        inputs = self.__tokenizer(text, return_tensors="np")
        current_input_ids = inputs["input_ids"]  # shape: (1, seq_len)
        
        print(f"Initial input_ids shape: {current_input_ids.shape}")
        print(f"Initial tokens: {current_input_ids[0].tolist()}")
        
        # 生成多个新token
        generated_tokens = []
        
        for step in range(max_new_tokens):
            print(f"\n=== Generation Step {step + 1} ===")
            
            # 更新attention_mask (全1表示所有位置都要关注)
            current_attention_mask = np.ones_like(current_input_ids)
            
            # 更新position_ids (序列位置从0开始)
            seq_length = current_input_ids.shape[1]
            current_position_ids = np.arange(seq_length).reshape(1, -1).astype(np.int64)
            
            # beam_idx保持不变
            batch_size = current_input_ids.shape[0]
            beam_idx = np.zeros(batch_size, dtype=np.int32)
            
            # 准备推理输入字典
            input_dict = {}
            for input_layer in self.inputs:
                input_name = input_layer.any_name
                if "input_ids" in input_name:
                    input_dict[input_name] = current_input_ids
                elif "attention_mask" in input_name:
                    input_dict[input_name] = current_attention_mask
                elif "position_ids" in input_name:
                    input_dict[input_name] = current_position_ids
                elif "beam_idx" in input_name:
                    input_dict[input_name] = beam_idx
            
            print(f"Step {step + 1} input shapes: input_ids={current_input_ids.shape}, "
                  f"attention_mask={current_attention_mask.shape}, "
                  f"position_ids={current_position_ids.shape}")
            
            # 执行推理
            infer_request = self.compiled_model.create_infer_request()
            result = infer_request.infer(input_dict)
            output = result[self.outputs[0]]
            
            # 获取最后一个位置的logits并预测下一个token
            last_token_logits = output[0, -1, :]  # shape: (vocab_size,)
            next_token_id = np.argmax(last_token_logits)
            
            # 解码新token
            next_token = self.__tokenizer.decode([next_token_id])
            generated_tokens.append(next_token_id)
            
            print(f"Step {step + 1} predicted token: '{next_token}' (id: {next_token_id})")
            
            # 将新token追加到输入序列中，准备下一次推理
            current_input_ids = np.concatenate([
                current_input_ids, 
                np.array([[next_token_id]], dtype=np.int64)
            ], axis=1)
            
            print(f"Updated input_ids shape: {current_input_ids.shape}")
            
            # 检查是否是结束token (EOS)
            if next_token_id == self.__tokenizer.eos_token_id:
                print(f"Generated EOS token, stopping generation")
                break
        
        # 解码所有生成的tokens
        generated_text = self.__tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"\nGenerated tokens: {generated_tokens}")
        print(f"Generated text: '{generated_text}'")
        
        return f"Native OpenVINO inference result for: {desc} -> Generated: '{generated_text}'"

def unit_test_native_ov(model_path, device):
    model = Model_Qwen3_1_7B_Native_OV(model_path, device)
    for i in range(4):
        t1 = time.time()
        res = model.infer("今天天气不错，和是出去游山玩水哈！")
        t2 = time.time()
        print(f" == {i} time = {t2-t1:.2f} s.")
    print(f"== Result: {res}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Qwen3-1.7B Native OpenVINO model inference')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Model path, e.g.: /home/wy/models/Qwen3-1.7B-int8-ov')
    parser.add_argument('-d', '--device', type=str, default='GPU.1',
                        help='Device name for inference (default: GPU.1)')
    
    args = parser.parse_args()
    unit_test_native_ov(args.model_path, args.device)
