#include "llaisys/models/qwen2.h"
#include "llaisys_tensor.hpp"

#include <cmath>
#include <iostream>
#include <vector>

#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../tensor/tensor.hpp"

using namespace llaisys;

struct LlaisysQwen2KVCache {
    std::vector<tensor_t> attn_kv_cache;
    size_t max_len;
    size_t cache_pos; // Current position in KV cache

    LlaisysQwen2KVCache(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, size_t max_len)
        : max_len(max_len), cache_pos(0) {
        // Initialize KV cache
        for (size_t j = 0; j < meta->nlayer; ++j) {
            // For each layer, create K and V cache tensors
            tensor_t k_cache = llaisys::Tensor::create({1, max_len, meta->nkvh, meta->dh}, meta->dtype, device, 0);
            tensor_t v_cache = llaisys::Tensor::create({1, max_len, meta->nkvh, meta->dh}, meta->dtype, device, 0);
            attn_kv_cache.push_back(k_cache);
            attn_kv_cache.push_back(v_cache);
        }
    }

    ~LlaisysQwen2KVCache() {
        // KV cache tensors will be automatically destroyed when vector is destroyed
    }
};

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;
    llaisysDeviceType_t device;

    // Workspace tensors
    tensor_t hidden_states;
    tensor_t norm_output;
    tensor_t attn_q;
    tensor_t attn_k;
    tensor_t attn_v;
    tensor_t attn_rotated_q;
    tensor_t attn_rotated_k;
    tensor_t attn_output;
    tensor_t mlp_gate;
    tensor_t mlp_up;
    tensor_t mlp_swiglu;
    tensor_t mlp_down;
    tensor_t logits;
    tensor_t max_val; // For storing maximum value (required by argmax)
    tensor_t next_token;

    LlaisysQwen2Model(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device) : meta(*meta), device(device) {
        // Initialize weights arrays
        weights.attn_norm_w = new llaisysTensor_t[meta->nlayer];
        weights.attn_q_w = new llaisysTensor_t[meta->nlayer];
        weights.attn_q_b = new llaisysTensor_t[meta->nlayer];
        weights.attn_k_w = new llaisysTensor_t[meta->nlayer];
        weights.attn_k_b = new llaisysTensor_t[meta->nlayer];
        weights.attn_v_w = new llaisysTensor_t[meta->nlayer];
        weights.attn_v_b = new llaisysTensor_t[meta->nlayer];
        weights.attn_o_w = new llaisysTensor_t[meta->nlayer];
        weights.mlp_norm_w = new llaisysTensor_t[meta->nlayer];
        weights.mlp_gate_w = new llaisysTensor_t[meta->nlayer];
        weights.mlp_up_w = new llaisysTensor_t[meta->nlayer];
        weights.mlp_down_w = new llaisysTensor_t[meta->nlayer];

        // Initialize workspace tensors
        hidden_states = llaisys::Tensor::create({1, 1, meta->hs}, meta->dtype, device, 0);
        norm_output = llaisys::Tensor::create({1, 1, meta->hs}, meta->dtype, device, 0);
        attn_q = llaisys::Tensor::create({1, 1, meta->nh, meta->dh}, meta->dtype, device, 0);
        attn_k = llaisys::Tensor::create({1, 1, meta->nkvh, meta->dh}, meta->dtype, device, 0);
        attn_v = llaisys::Tensor::create({1, 1, meta->nkvh, meta->dh}, meta->dtype, device, 0);
        attn_rotated_q = llaisys::Tensor::create({1, 1, meta->nh, meta->dh}, meta->dtype, device, 0);
        attn_rotated_k = llaisys::Tensor::create({1, 1, meta->nkvh, meta->dh}, meta->dtype, device, 0);
        attn_output = llaisys::Tensor::create({1, 1, meta->hs}, meta->dtype, device, 0);
        mlp_gate = llaisys::Tensor::create({1, 1, meta->di}, meta->dtype, device, 0);
        mlp_up = llaisys::Tensor::create({1, 1, meta->di}, meta->dtype, device, 0);
        mlp_swiglu = llaisys::Tensor::create({1, 1, meta->di}, meta->dtype, device, 0);
        mlp_down = llaisys::Tensor::create({1, 1, meta->hs}, meta->dtype, device, 0);
        logits = llaisys::Tensor::create({1, 1, meta->voc}, meta->dtype, device, 0);
        max_val = llaisys::Tensor::create({1}, meta->dtype, device, 0);
        next_token = llaisys::Tensor::create({1}, llaisysDataType_t::LLAISYS_DTYPE_I64, device, 0);
    }

    ~LlaisysQwen2Model() {
        // Free weights arrays
        delete[] weights.attn_norm_w;
        delete[] weights.attn_q_w;
        delete[] weights.attn_q_b;
        delete[] weights.attn_k_w;
        delete[] weights.attn_k_b;
        delete[] weights.attn_v_w;
        delete[] weights.attn_v_b;
        delete[] weights.attn_o_w;
        delete[] weights.mlp_norm_w;
        delete[] weights.mlp_gate_w;
        delete[] weights.mlp_up_w;
        delete[] weights.mlp_down_w;
    }
};

__C {
    LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
        // Default to device ID 0 if no valid device IDs provided
        int device_id = 0;
        if (ndevice > 0 && device_ids != nullptr) {
            device_id = device_ids[0];
        }

        // Set current device with proper error handling
        llaisys::core::context().setDevice(device, device_id);

        return new LlaisysQwen2Model(meta, device);
    }

    void llaisysQwen2ModelDestroy(LlaisysQwen2Model * model) {
        delete model;
    }

    LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model * model) {
        return &model->weights;
    }

    LlaisysQwen2KVCache *llaisysQwen2KVCacheCreate(LlaisysQwen2Model * model, size_t max_len) {
        return new LlaisysQwen2KVCache(&model->meta, model->device, max_len);
    }

    void llaisysQwen2KVCacheDestroy(LlaisysQwen2KVCache * kvcache, size_t max_len) {
        delete kvcache;
    }

    void llaisysQwen2ModelResetCache(LlaisysQwen2Model * model) {
        // This function is kept for backward compatibility
    }

    int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model * model, int64_t *token_ids, size_t ntoken, LlaisysQwen2KVCache *kvcache, size_t past_len) {
        // Get model metadata
        const LlaisysQwen2Meta &meta = model->meta;
        LlaisysQwen2Weights &weights = model->weights;

        // Initialize hidden states with embedding
        tensor_t input_ids = llaisys::Tensor::create({1, ntoken}, llaisysDataType_t::LLAISYS_DTYPE_I64, model->device, 0);
        input_ids->load(token_ids);
        ops::embedding(model->hidden_states, input_ids, weights.in_embed->tensor);

        // Process each transformer layer
        for (size_t i = 0; i < meta.nlayer; ++i) {
            // Self-attention block
            ops::rms_norm(model->norm_output, model->hidden_states, weights.attn_norm_w[i]->tensor, meta.epsilon);

            // Compute Q projection
            ops::linear(model->attn_q, model->norm_output, weights.attn_q_w[i]->tensor, weights.attn_q_b[i]->tensor);

            // Compute K projection
            ops::linear(model->attn_k, model->norm_output, weights.attn_k_w[i]->tensor, weights.attn_k_b[i]->tensor);

            // Compute V projection
            ops::linear(model->attn_v, model->norm_output, weights.attn_v_w[i]->tensor, weights.attn_v_b[i]->tensor);

            // Create position ids tensor
            tensor_t pos_ids = llaisys::Tensor::create({1, 1}, llaisysDataType_t::LLAISYS_DTYPE_I64, model->device, 0);
            int64_t pos = past_len + ntoken - 1;
            pos_ids->load(&pos);

            // Apply RoPE
            ops::rope(model->attn_rotated_q, model->attn_q, pos_ids, meta.theta);
            ops::rope(model->attn_rotated_k, model->attn_k, pos_ids, meta.theta);

            // Self-attention with KV cache
            float scale = 1.0f / std::sqrt(meta.dh);
            ops::self_attention(model->attn_output, model->attn_rotated_q, model->attn_rotated_k, model->attn_v, scale);

            // Attention output projection
            ops::linear(model->attn_output, model->attn_output, weights.attn_o_w[i]->tensor, nullptr);

            // Residual connection (hidden_states += attn_output)
            ops::add(model->hidden_states, model->hidden_states, model->attn_output);

            // MLP block
            ops::rms_norm(model->norm_output, model->hidden_states, weights.mlp_norm_w[i]->tensor, meta.epsilon);

            // MLP layers
            ops::linear(model->mlp_gate, model->norm_output, weights.mlp_gate_w[i]->tensor, nullptr);
            ops::linear(model->mlp_up, model->norm_output, weights.mlp_up_w[i]->tensor, nullptr);
            ops::swiglu(model->mlp_swiglu, model->mlp_gate, model->mlp_up);
            ops::linear(model->mlp_down, model->mlp_swiglu, weights.mlp_down_w[i]->tensor, nullptr);

            // Residual connection (hidden_states += mlp_down)
            ops::add(model->hidden_states, model->hidden_states, model->mlp_down);
        }

        // Final normalization
        ops::rms_norm(model->norm_output, model->hidden_states, weights.out_norm_w->tensor, meta.epsilon);

        // Output layer
        ops::linear(model->logits, model->norm_output, weights.out_embed->tensor, nullptr);

        // Get argmax of logits to get next token
        // Use view to reshape logits from [1, 1, vocab] to [vocab]
        tensor_t logits_1d = model->logits->view({static_cast<size_t>(meta.voc)});
        ops::argmax(model->next_token, model->max_val, logits_1d);

        // Extract the next token
        int64_t *next_token_ptr = reinterpret_cast<int64_t *>(model->next_token->data());
        int64_t next_token = next_token_ptr[0];

        return next_token;
    }
}