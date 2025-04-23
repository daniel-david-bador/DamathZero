module;

#include <torch/torch.h>

export module damathzero:model;

import std;
import alphazero;
import alphazero.model.transformer;

using namespace AZ::Models::Transformer;
namespace nn = torch::nn;

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
    encoder = register_module(
        "encoder",
        std::make_shared<Encoder>(
            config.num_blocks, config.embedding_dim, config.num_attention_head,
            config.mlp_hidden_size, config.mlp_dropout_prob));

    embedding = register_module(
        "embedding",
        std::make_shared<Embedding>(config.embedding_dim, /*feature_width=*/8,
                                    /*feature_height=*/8, /*num_channels=*/6));

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

  std::shared_ptr<Encoder> encoder{nullptr};
  std::shared_ptr<Embedding> embedding{nullptr};

  nn::Linear wdl_head{nullptr};
  nn::Linear policy_head{nullptr};
};

static_assert(AZ::Concepts::Model<Model>);
