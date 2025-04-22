module;

#include <torch/torch.h>

import std;

export module alphazero:model;

namespace AZ {

namespace Concepts {

export template <typename M>
concept Model = std::is_base_of_v<torch::nn::Module, M> and
                requires(M m, typename M::Config config, torch::Tensor x) {
                  typename M::Config;

                  { M(config) } -> std::same_as<M>;

                  std::same_as<decltype(m.config), typename M::Config>;

                  {
                    m.forward(x)
                  } -> std::same_as<std::tuple<torch::Tensor, torch::Tensor>>;
                };
}  // namespace Concepts

template <Concepts::Model Model>
auto clone_model(std::shared_ptr<Model> model, torch::DeviceType device)
    -> std::shared_ptr<Model> {
  auto cloned = std::make_shared<Model>(model->config);
  auto data = std::string();

  {
    std::ostringstream oss;
    torch::serialize::OutputArchive out_archive;
    model->to(torch::kCPU);
    model->save(out_archive);
    out_archive.save_to(oss);
    data = oss.str();
  }

  {
    std::istringstream iss(data);
    torch::serialize::InputArchive in_archive;
    in_archive.load_from(iss);
    cloned->load(in_archive);
    cloned->to(device);
  }

  return cloned;
}

template <Concepts::Model Model>
auto save_model(std::shared_ptr<Model> model, int checkpoint) -> void {
  torch::serialize::OutputArchive output_model_archive;
  model->to(torch::kCPU);
  model->save(output_model_archive);
  output_model_archive.save_to(std::format("models/model_{}.pt", checkpoint));
};

template <Concepts::Model Model>
auto read_model(int checkpoint, torch::DeviceType device,
                typename Model::Config config) -> std::shared_ptr<Model> {
  torch::serialize::InputArchive input_archive;
  input_archive.load_from(std::format("models/model_{}.pt", checkpoint));
  auto model = std::make_shared<Model>(config);
  model->load(input_archive);
  model->to(device);
  return model;
}

}  // namespace AZ
