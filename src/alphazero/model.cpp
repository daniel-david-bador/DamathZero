module;

#include <torch/torch.h>

export module az:model;

import std;

namespace az {

namespace concepts {

export template <typename M>
concept Model = std::is_base_of_v<torch::nn::Module, M> and
                requires(M m, typename M::Config config, torch::Tensor x) {
                  { M(config) } -> std::same_as<M>;

                  { m.config } -> std::same_as<typename M::Config&>;

                  {
                    m.forward(x)
                  } -> std::same_as<std::tuple<torch::Tensor, torch::Tensor>>;
                };
}  // namespace concepts

namespace utils {

export template <concepts::Model Model>
auto clone_model(std::shared_ptr<Model> model) -> std::shared_ptr<Model> {
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
    cloned->to(torch::kCPU);
  }

  return cloned;
}

export template <concepts::Model Model>
auto save_model(std::shared_ptr<Model> model, std::string_view path) -> void {
  torch::serialize::OutputArchive output_model_archive;
  model->to(torch::kCPU);
  model->save(output_model_archive);
  output_model_archive.save_to(std::string(path));
};

export template <concepts::Model Model>
auto load_model(std::string_view path, typename Model::Config config)
    -> std::shared_ptr<Model> {
  torch::serialize::InputArchive input_archive;
  input_archive.load_from(std::string(path));
  auto model = std::make_shared<Model>(config);
  model->load(input_archive);
  model->to(torch::kCPU);
  return model;
}

}  // namespace utils

}  // namespace az
