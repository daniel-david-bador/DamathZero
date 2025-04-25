module;

#include <torch/torch.h>

export module az:transformer;

import std;

namespace az::models {

namespace nn = torch::nn;
namespace F = nn::functional;

// Position vs Vector Embedding
export struct Embedding : public nn::Module {
  Embedding(int32_t embedding_dim, int32_t feature_width,
            int32_t feature_height, int32_t num_channels) {
    auto opts =
        nn::LinearOptions(feature_width * feature_height, embedding_dim);
    projection = register_module("projection", nn::Linear(opts));

    // Create positional embedding add one for the CLS token.
    positional_embedding =
        register_parameter("positional_encoding",
                           torch::randn({1, num_channels + 1, embedding_dim}));
    cls_token =
        register_parameter("cls_token", torch::randn({1, 1, embedding_dim}));
  }

  auto forward(torch::Tensor x) -> torch::Tensor {
    // we expect the tensor to have a shape of (N, H, W, C)
    assert(x.sizes().size() == 4);

    // (N, H, W, C) -> (N, C, H, W)
    x = x.permute({0, 3, 1, 2});
    // (N, C, H, W) -> (N, C, H * W)
    x = x.flatten(2);
    // (N, C, H * W) -> (N, C, embedding_dim)
    x = projection->forward(x);

    auto batch_size = x.size(0);
    auto cls = cls_token.expand({batch_size, -1, -1});

    // Concatanete the cls token at the beginning
    // (N, C, embedding_dim) -> (N, C+1, embedding_dim)
    x = torch::cat({cls, x}, /*dim=*/1);
    return x + positional_embedding;
  };

  nn::Linear projection{nullptr};
  torch::Tensor positional_embedding;
  torch::Tensor cls_token;
};

struct Attention : public nn::Module {
  Attention(int32_t embedding_dim, int32_t num_heads) {
    auto attention_ops =
        nn::MultiheadAttentionOptions(embedding_dim, num_heads);

    attention =
        register_module("attention", nn::MultiheadAttention(attention_ops));

    query = register_module("query", nn::Linear(embedding_dim, embedding_dim));
    key = register_module("key", nn::Linear(embedding_dim, embedding_dim));
    value = register_module("value", nn::Linear(embedding_dim, embedding_dim));
  }

  auto forward(torch::Tensor x, bool output_attention = false)
      -> std::tuple<torch::Tensor, std::optional<torch::Tensor>> {
    // x is of shape (N, L, H, W) but MultiheadAttention expects (L, N, H, W)
    auto x_t = x.transpose(0, 1);

    auto q = query->forward(x_t);
    auto k = key->forward(x_t);
    auto v = value->forward(x_t);

    auto [out, probs] = attention->forward(q, k, v);

    out = out.transpose(0, 1);

    if (not output_attention) {
      return {out, std::nullopt};
    }

    return {out, probs.transpose(0, 1)};
  }

  nn::MultiheadAttention attention{nullptr};
  nn::Linear query{nullptr};
  nn::Linear key{nullptr};
  nn::Linear value{nullptr};
};

struct MLP : public nn::Module {
  MLP(int32_t embedding_dim, int32_t hidden_size, float32_t dropout_prob) {
    layer1 = register_module("layer1", nn::Linear(embedding_dim, hidden_size));
    layer2 = register_module("layer2", nn::Linear(hidden_size, embedding_dim));
    dropout = register_module("dropout", nn::Dropout(dropout_prob));
  }

  auto forward(torch::Tensor x) -> torch::Tensor {
    x = layer1->forward(x);
    x = F::relu(x);
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

    layer_norm1 = register_module("layer_norm1", nn::LayerNorm(opts));
    layer_norm2 = register_module("layer_norm2", nn::LayerNorm(opts));

    mlp =
        std::make_shared<MLP>(embedding_dim, mlp_hidden_size, mlp_dropout_prob);

    attention = std::make_shared<Attention>(embedding_dim, num_attention_heads);
  }

  auto forward(torch::Tensor x, bool output_attention = false)
      -> std::tuple<torch::Tensor, std::optional<torch::Tensor>> {
    auto [attention_out, attention_probs] =
        attention->forward(layer_norm1->forward(x));
    x = x + attention_out;
    auto mlp_out = mlp->forward(layer_norm2->forward(x));
    x = x + mlp_out;

    if (not output_attention) {
      return {x, std::nullopt};
    }

    return {x, attention_probs};
  }

  nn::LayerNorm layer_norm1{nullptr};
  nn::LayerNorm layer_norm2{nullptr};

  std::shared_ptr<MLP> mlp;
  std::shared_ptr<Attention> attention;
};

export struct TransformerEncoder : public nn::Module {
  TransformerEncoder(int32_t num_blocks, int32_t embedding_dim,
                     int32_t num_attention_heads, int32_t mlp_hidden_size,
                     float32_t mlp_dropout_prob) {
    assert(embedding_dim % num_attention_heads == 0);

    blocks = register_module("blocks", nn::ModuleList());
    for (auto _ : std::views::iota(0, num_blocks)) {
      auto block = std::make_shared<Block>(embedding_dim, num_attention_heads,
                                           mlp_hidden_size, mlp_dropout_prob);
      blocks->push_back(block);
    }
  }

  auto forward(torch::Tensor x, bool output_attention = false)
      -> std::tuple<torch::Tensor, std::optional<torch::Tensor>> {
    auto attentions = std::vector<torch::Tensor>();

    for (auto& block : *blocks) {
      auto [out, attention] = block->as<Block>()->forward(x, output_attention);
      x = std::move(out);
      if (output_attention) {
        attentions.push_back(*attention);
      }
    }

    x = x.select(1, 0);

    if (not output_attention) {
      return {x, std::nullopt};
    }

    return {x, torch::stack(attentions, 1)};
  }

  nn::ModuleList blocks{nullptr};
};

}  // namespace az::models
