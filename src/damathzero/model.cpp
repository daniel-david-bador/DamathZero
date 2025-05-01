module;

#include <torch/torch.h>

export module dz:model;

import az;
import std;

namespace dz {

namespace nn = torch::nn;
namespace F = nn::functional;

// Patch and positional embeddings
struct Embedding : public nn::Module {
  Embedding(int32_t num_cls_tokens, int32_t feature_height,
            int32_t feature_width, int32_t embedding_dim) {
    projection =
        register_module("projection", nn::Linear(feature_width, embedding_dim));

    positional_embedding = register_parameter(
        "positional_embedding",
        torch::randn({1, feature_height + num_cls_tokens, embedding_dim}));

    cls_tokens = register_parameter(
        "cls_tokens", torch::randn({1, num_cls_tokens, embedding_dim}));

    layer_norm = register_module(
        "layer_norm", nn::LayerNorm(nn::LayerNormOptions({embedding_dim})));
  }

  auto forward(torch::Tensor x) -> torch::Tensor {
    // we expect the tensor to have a shape of (N, H, W)
    assert(x.sizes().size() == 3);

    const auto N = x.size(0);

    x = projection->forward(x);
    auto cls = cls_tokens.expand({N, -1, -1});
    x = torch::cat({cls, x}, 1);
    x = x + positional_embedding;
    return layer_norm->forward(x);
  };

  torch::Tensor positional_embedding;
  torch::Tensor cls_tokens;

  nn::Linear projection{nullptr};
  nn::LayerNorm layer_norm{nullptr};
};

struct MLP : public nn::Module {
  MLP(int32_t embedding_dim, int32_t hidden_size, float32_t dropout_prob) {
    layer1 = register_module("layer1", nn::Linear(embedding_dim, hidden_size));
    layer2 = register_module("layer2", nn::Linear(hidden_size, embedding_dim));
    dropout = register_module("dropout", nn::Dropout(dropout_prob));
  }

  auto forward(torch::Tensor x) -> torch::Tensor {
    x = layer1->forward(x);
    x = F::gelu(x);
    x = layer2->forward(x);
    x = dropout->forward(x);
    return x;
  };

  nn::Linear layer1{nullptr};
  nn::Linear layer2{nullptr};
  nn::Dropout dropout{nullptr};
};

struct Block : public nn::Module {
  Block(int32_t embedding_dim, int32_t num_attention_heads,
        int32_t mlp_hidden_size, float32_t mlp_dropout_prob) {
    auto opts = nn::LayerNormOptions({embedding_dim});
    auto attention_ops =
        nn::MultiheadAttentionOptions(embedding_dim, num_attention_heads);

    attention =
        register_module("attention", nn::MultiheadAttention(attention_ops));

    layer_norm1 = register_module("layer_norm1", nn::LayerNorm(opts));
    layer_norm2 = register_module("layer_norm2", nn::LayerNorm(opts));

    mlp = register_module("mlp",
                          std::make_shared<MLP>(embedding_dim, mlp_hidden_size,
                                                mlp_dropout_prob));
  }

  auto forward(torch::Tensor x, bool output_attention = false)
      -> std::tuple<torch::Tensor, std::optional<torch::Tensor>> {
    x = layer_norm1->forward(x);

    auto [attention_out, attention_probs] = attention->forward(x, x, x);
    x = x + attention_out;

    x = layer_norm2->forward(x);
    auto mlp_out = mlp->forward(x);
    x = x + mlp_out;

    if (not output_attention) {
      return {x, std::nullopt};
    }

    return {x, attention_probs};
  }

  nn::MultiheadAttention attention{nullptr};

  nn::LayerNorm layer_norm1{nullptr};
  nn::LayerNorm layer_norm2{nullptr};

  std::shared_ptr<MLP> mlp;
};

struct Encoder : public nn::Module {
  Encoder(int32_t num_cls_tokens, int32_t num_blocks, int32_t embedding_dim,
          int32_t num_attention_heads, int32_t mlp_hidden_size,
          float32_t mlp_dropout_prob)
      : num_cls_tokens(num_cls_tokens) {
    assert(embedding_dim % num_attention_heads == 0);

    blocks = register_module("blocks", nn::ModuleList());
    for (auto _ : std::views::iota(0, num_blocks)) {
      auto block = std::make_shared<Block>(embedding_dim, num_attention_heads,
                                           mlp_hidden_size, mlp_dropout_prob);
      blocks->push_back(std::move(block));
    }

    layer_norm = register_module(
        "layer_norm", nn::LayerNorm(nn::LayerNormOptions({embedding_dim})));
  }

  auto forward(torch::Tensor x, bool output_attention = false)
      -> std::tuple<torch::Tensor, std::optional<torch::Tensor>> {
    auto attentions = std::vector<torch::Tensor>();

    // X has a shape of (N, L, embedding_dim)
    // but the multihead attention layers in all the blocks expects it to be
    // (L, N, embedding_dim)
    x = x.transpose(0, 1);

    for (auto& block : *blocks) {
      auto [out, attention] = block->as<Block>()->forward(x, output_attention);
      x = std::move(out);
      if (output_attention)
        attentions.push_back(*attention);
    }

    x = layer_norm->forward(x);
    // Tranpose back
    x = x.transpose(0, 1);
    x = x.slice(1, 0, num_cls_tokens);
    // Flatten for the feed forward networks
    x = x.flatten(1);

    if (not output_attention) {
      return {x, std::nullopt};
    }

    return {x, torch::stack(attentions, 1)};
  }

  int32_t num_cls_tokens;

  nn::LayerNorm layer_norm{nullptr};
  nn::ModuleList blocks{nullptr};
};

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
    const auto feature_width = 32;
    const auto feature_height = 11;
    const auto num_cls_tokens = 8;

    // assert(feature_width % patch_size == 0);

    encoder = register_module(
        "encoder", std::make_shared<Encoder>(
                       num_cls_tokens, config.num_blocks, config.embedding_dim,
                       config.num_attention_head, config.mlp_hidden_size,
                       config.mlp_dropout_prob));

    embedding = register_module(
        "embedding",
        std::make_shared<Embedding>(num_cls_tokens, feature_width,
                                    feature_height, config.embedding_dim));

    wdl_head = register_module(
        "wdl_head", nn::Linear(num_cls_tokens * config.embedding_dim, 3));
    policy_head = register_module(
        "policy_head",
        nn::Linear(num_cls_tokens * config.embedding_dim, config.action_size));
  }

  auto forward(torch::Tensor x) -> std::tuple<torch::Tensor, torch::Tensor> {
    x = embedding->forward(x);
    auto [out, _] = encoder->forward(x, /*output_attention=*/false);

    auto wdl = F::softmax(wdl_head->forward(out), 1);
    auto policy = policy_head->forward(out);

    assert(wdl.size(1) == 3);
    assert(policy.size(1) == config.action_size);
    return {wdl, policy};
  }

  Config config;

  std::shared_ptr<Encoder> encoder{nullptr};
  std::shared_ptr<Embedding> embedding{nullptr};

  nn::Linear feed_forward{nullptr};
  nn::Linear wdl_head{nullptr};
  nn::Linear policy_head{nullptr};
};

static_assert(az::concepts::Model<Model>);

}  // namespace dz
