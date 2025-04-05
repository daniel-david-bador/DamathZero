module;

#include <torch/torch.h>

import std;

export module alphazero:network;

namespace AZ {

namespace Concepts {

export template <typename N>
concept Network = std::is_base_of_v<torch::nn::Module, N>;

}

}  // namespace AZ
