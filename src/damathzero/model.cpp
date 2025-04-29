module;

#include <torch/torch.h>

export module dz:model;

import :vit;

import az;
import std;

namespace dz {

namespace nn = torch::nn;
namespace F = nn::functional;

export struct Model : torch::nn::Module {
  struct Config {
    int32_t action_size;
    int32_t num_blocks;
    int32_t num_attention_head;
    int32_t embedding_dim;
    int32_t mlp_hidden_size;
    float32_t mlp_dropout_prob;
  };

  Model(Config config) : config(config) {
    const auto patch_size = 2;
    const auto feature_width = 8;

    assert(feature_width % patch_size == 0);

    encoder = register_module(
        "encoder",
        std::make_shared<vit::Encoder>(
            config.num_blocks, config.embedding_dim, config.num_attention_head,
            config.mlp_hidden_size, config.mlp_dropout_prob));

    embedding = register_module(
        "embedding", std::make_shared<vit::Embedding>(feature_width, patch_size,
                                                      config.embedding_dim,
                                                      /*num_channels=*/10));

    wdl_head = register_module("wdl_head", nn::Linear(config.embedding_dim, 3));
    policy_head = register_module(
        "policy_head", nn::Linear(config.embedding_dim, config.action_size));
  }

  auto forward(torch::Tensor x) -> std::tuple<torch::Tensor, torch::Tensor> {
    namespace F = torch::nn::functional;

    x = embedding->forward(x);
    auto [out, _] = encoder->forward(x, /*output_attention=*/false);

    auto wdl = F::softmax(wdl_head->forward(out), 1);
    auto policy = policy_head->forward(out);
    return {wdl, policy};
  }

  Config config;

  std::shared_ptr<vit::Encoder> encoder{nullptr};
  std::shared_ptr<vit::Embedding> embedding{nullptr};

  nn::Linear wdl_head{nullptr};
  nn::Linear policy_head{nullptr};
};

static_assert(az::concepts::Model<Model>);

}  // namespace dz
