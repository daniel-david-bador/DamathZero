#pragma once

#include <torch/torch.h>

#include <cstdint>

#include "alphazero/model.hpp"
namespace dz {

namespace nn = torch::nn;

// Patch and positional embeddings
struct Embedding : public nn::Module {
  Embedding(int32_t num_cls_tokens, int32_t feature_height,
            int32_t feature_width, int32_t embedding_dim);

  auto forward(torch::Tensor x) -> torch::Tensor;

  torch::Tensor cls_tokens;
  torch::Tensor positional_embedding;

  nn::Linear projection{nullptr};
  nn::LayerNorm layer_norm{nullptr};
};

struct MultilayerPerceptron : public nn::Module {
  MultilayerPerceptron(int32_t embedding_dim, int32_t hidden_size,
                       float32_t dropout_prob);
  auto forward(torch::Tensor x) -> torch::Tensor;

  nn::Linear layer1{nullptr};
  nn::Linear layer2{nullptr};
  nn::Dropout dropout{nullptr};
};

struct Block : public nn::Module {
  Block(int32_t embedding_dim, int32_t num_attention_heads,
        int32_t mlp_hidden_size, float32_t mlp_dropout_prob);

  auto forward(torch::Tensor x, bool output_attention = false)
      -> std::tuple<torch::Tensor, std::optional<torch::Tensor>>;

  nn::MultiheadAttention attention{nullptr};

  nn::LayerNorm layer_norm1{nullptr};
  nn::LayerNorm layer_norm2{nullptr};

  std::shared_ptr<MultilayerPerceptron> mlp;
};

struct Encoder : public nn::Module {
  Encoder(int32_t num_cls_tokens, int32_t num_blocks, int32_t embedding_dim,
          int32_t num_attention_heads, int32_t mlp_hidden_size,
          float32_t mlp_dropout_prob);

  auto forward(torch::Tensor x, bool output_attention = false)
      -> std::tuple<torch::Tensor, std::optional<torch::Tensor>>;

  int32_t num_cls_tokens;

  nn::LayerNorm layer_norm{nullptr};
  nn::ModuleList blocks{nullptr};
};

struct Model : torch::nn::Module {
  struct Config {
    int32_t action_size;
    int32_t num_blocks;
    int32_t num_attention_head;
    int32_t embedding_dim;
    int32_t mlp_hidden_size;
    float32_t mlp_dropout_prob;
    torch::Device device;
  };

  Model(Config config);

  auto forward(torch::Tensor x) -> std::tuple<torch::Tensor, torch::Tensor>;

  Config config;

  std::shared_ptr<Encoder> encoder{nullptr};
  std::shared_ptr<Embedding> embedding{nullptr};

  nn::Linear wdl_head{nullptr};
  nn::Linear policy_head{nullptr};
};

static_assert(az::concepts::Model<Model>);

}  // namespace dz
