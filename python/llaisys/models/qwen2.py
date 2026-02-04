from typing import Sequence, List
from ..libllaisys import LIB_LLAISYS, DeviceType, Qwen2MetaCStruct, Qwen2WeightsCStruct, DataType, Qwen2WeightsNaming
from ..tensor import Tensor

from pathlib import Path
import safetensors
import numpy as np
import ctypes
import json
import torch


class Qwen2:
    def __init__(self, model_path: str, device: DeviceType = DeviceType.CPU):
        self.model_path = Path(model_path)
        self.device = device
        self.model_ptr = None
        self.kvcache_ptr = None
        
        # Load model metadata
        self.meta = self._load_metadata()
        
        # Create model
        self._create_model()
        
        # Create KV cache
        self._create_kvcache()
        
        # Load model weights
        self._load_weights()
        
    def _load_metadata(self) -> Qwen2MetaCStruct:
        # Load config.json to get actual model parameters
        config_path = self.model_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # Fallback config
            config = {
                "torch_dtype": "float32",
                "num_hidden_layers": 24,
                "hidden_size": 2048,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "intermediate_size": 5632,
                "max_position_embeddings": 8192,
                "vocab_size": 151936,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "eos_token_id": 151643
            }
        
        # Convert torch dtype string to numpy dtype
        torch_to_numpy_dtype = {
            "float32": np.float32,
            "float16": np.float16,
            "bfloat16": np.float32,  # Using float32 as approximation for bfloat16
        }
        config["torch_dtype"] = torch_to_numpy_dtype.get(config.get("torch_dtype", "float32"), np.float32)
        
        meta = Qwen2MetaCStruct(config)
        return meta
        
    def _create_model(self):
        # Create model instance
        # 根据错误信息，尝试参数顺序 (meta, nlayer, naming, dtype)
        
        # 创建命名实例
        naming = Qwen2WeightsNaming()
        
        # 创建临时的state_dict用于初始化
        temp_state_dict = {}
        
        # 添加必要的权重键到临时字典
        temp_state_dict[naming.input_embed()] = torch.empty(0, dtype=torch.float32)  # 占位符
        temp_state_dict[naming.output_embed()] = torch.empty(0, dtype=torch.float32)  # 占位符
        temp_state_dict[naming.output_norm()] = torch.empty(0, dtype=torch.float32)  # 占位符
        
        # 为每一层添加权重键
        for i in range(self.meta.nlayer):
            temp_state_dict[naming.attn_norm(i)] = torch.empty(0, dtype=torch.float32)  # 占位符
            temp_state_dict[naming.attn_q(i)] = torch.empty(0, dtype=torch.float32)  # 占位符
            temp_state_dict[naming.attn_k(i)] = torch.empty(0, dtype=torch.float32)  # 占位符
            temp_state_dict[naming.attn_v(i)] = torch.empty(0, dtype=torch.float32)  # 占位符
            temp_state_dict[naming.attn_o(i)] = torch.empty(0, dtype=torch.float32)  # 占位符
            temp_state_dict[naming.attn_q_b(i)] = torch.empty(0, dtype=torch.float32)  # 占位符
            temp_state_dict[naming.attn_k_b(i)] = torch.empty(0, dtype=torch.float32)  # 占位符
            temp_state_dict[naming.attn_v_b(i)] = torch.empty(0, dtype=torch.float32)  # 占位符
            temp_state_dict[naming.mlp_norm(i)] = torch.empty(0, dtype=torch.float32)  # 占位符
            temp_state_dict[naming.gate(i)] = torch.empty(0, dtype=torch.float32)  # 占位符
            temp_state_dict[naming.up(i)] = torch.empty(0, dtype=torch.float32)  # 占位符
            temp_state_dict[naming.down(i)] = torch.empty(0, dtype=torch.float32)  # 占位符
        
        # 尝试参数顺序 (meta, nlayer, naming, dtype)
        temp_weights = Qwen2WeightsCStruct(self.meta, self.meta.nlayer, naming, torch.float32)
        
        # 设置设备ID
        device_ids = (ctypes.c_int * 1)(0)  # 设备ID为0，即CPU
        ndevice = 1
        
        self.model_ptr = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self.meta),
            ctypes.byref(temp_weights),  # 权重结构
            self.device.value,           # 设备类型
            ndevice,                     # ndevice
            device_ids                   # device_ids
        )
        
        if not self.model_ptr:
            raise RuntimeError("Failed to create Qwen2 model")
    
    def _create_kvcache(self):
        # Create KV cache for the model
        max_len = 1024  # Maximum sequence length for KV cache
        self.kvcache_ptr = LIB_LLAISYS.llaisysQwen2KVCacheCreate(self.model_ptr, max_len)
        
        if not self.kvcache_ptr:
            raise RuntimeError("Failed to create KV cache")
            
    def _load_weights(self):
        # Get weights pointer
        weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self.model_ptr)
        weights = weights_ptr.contents
        
        # Create naming instance to map weight names
        naming = Qwen2WeightsNaming()
        
        # Load weights from safetensors files and assign them
        for file in sorted(self.model_path.glob("*.safetensors")):
            data = safetensors.safe_open(file, framework="numpy", device="cpu")
            for name in data.keys():
                numpy_tensor = data.get_tensor(name)
                
                # Convert numpy tensor to llaisys tensor
                tensor = Tensor.from_numpy(numpy_tensor, self.device)
                
                # Assign tensor to the appropriate weight
                self._assign_weight(weights, name, tensor, naming)
                
    def _assign_weight(self, weights: Qwen2WeightsCStruct, name: str, tensor: Tensor, naming: Qwen2WeightsNaming):
        # Map weight names to their respective pointers using the naming convention
        if name == naming.input_embed():  # "model.embed_tokens.weight"
            weights.in_embed = tensor.lib_tensor()
        elif name == naming.output_embed():  # "lm_head.weight"
            weights.out_embed = tensor.lib_tensor()
        elif name == naming.output_norm():  # "model.norm.weight"
            weights.out_norm_w = tensor.lib_tensor()
        else:
            # Check for layer-specific weights
            for i in range(self.meta.nlayer):
                if name == naming.attn_norm(i):  # "model.layers.{i}.input_layernorm.weight"
                    weights.attn_norm_w[i] = tensor.lib_tensor()
                    break
                elif name == naming.attn_q(i):  # "model.layers.{i}.self_attn.q_proj.weight"
                    weights.attn_q_w[i] = tensor.lib_tensor()
                    break
                elif name == naming.attn_k(i):  # "model.layers.{i}.self_attn.k_proj.weight"
                    weights.attn_k_w[i] = tensor.lib_tensor()
                    break
                elif name == naming.attn_v(i):  # "model.layers.{i}.self_attn.v_proj.weight"
                    weights.attn_v_w[i] = tensor.lib_tensor()
                    break
                elif name == naming.attn_o(i):  # "model.layers.{i}.self_attn.o_proj.weight"
                    weights.attn_o_w[i] = tensor.lib_tensor()
                    break
                elif name == naming.attn_q_b(i):  # "model.layers.{i}.self_attn.q_proj.bias"
                    weights.attn_q_b[i] = tensor.lib_tensor()
                    break
                elif name == naming.attn_k_b(i):  # "model.layers.{i}.self_attn.k_proj.bias"
                    weights.attn_k_b[i] = tensor.lib_tensor()
                    break
                elif name == naming.attn_v_b(i):  # "model.layers.{i}.self_attn.v_proj.bias"
                    weights.attn_v_b[i] = tensor.lib_tensor()
                    break
                elif name == naming.mlp_norm(i):  # "model.layers.{i}.post_attention_layernorm.weight"
                    weights.mlp_norm_w[i] = tensor.lib_tensor()
                    break
                elif name == naming.gate(i):  # "model.layers.{i}.mlp.gate_proj.weight"
                    weights.mlp_gate_w[i] = tensor.lib_tensor()
                    break
                elif name == naming.up(i):  # "model.layers.{i}.mlp.up_proj.weight"
                    weights.mlp_up_w[i] = tensor.lib_tensor()
                    break
                elif name == naming.down(i):  # "model.layers.{i}.mlp.down_proj.weight"
                    weights.mlp_down_w[i] = tensor.lib_tensor()
                    break
    
    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 100,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ) -> List[int]:
        # Convert inputs to list
        tokens = list(inputs)
        
        # Reset KV cache for new sequence
        LIB_LLAISYS.llaisysQwen2ModelResetCache(self.model_ptr)
        
        # Generate tokens
        for _ in range(max_new_tokens):
            # Prepare token ids
            token_ids = (ctypes.c_int64 * len(tokens))(*tokens)
            
            # Run inference
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self.model_ptr,
                token_ids,
                len(tokens),
                self.kvcache_ptr,
                len(tokens) - 1  # past_len: number of tokens already processed
            )
            
            # Append new token
            tokens.append(next_token)
            
            # Check if we reached the end token
            if next_token == self.meta.end_token:
                break
                
        return tokens
        
    def __del__(self):
        # Destroy KV cache
        if hasattr(self, "kvcache_ptr") and self.kvcache_ptr:
            LIB_LLAISYS.llaisysQwen2KVCacheDestroy(self.kvcache_ptr)
            self.kvcache_ptr = None
            
        # Destroy model instance
        if hasattr(self, "model_ptr") and self.model_ptr:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model_ptr)
            self.model_ptr = None