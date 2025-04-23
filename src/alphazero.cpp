module;

#include <torch/torch.h>

#include <indicators/dynamic_progress.hpp>
#include <indicators/progress_bar.hpp>

export module alphazero;

import std;

export import :model;
export import :arena;
export import :config;
export import :game;
export import :memory;
export import :mcts;
export import :node;
export import :storage;

namespace opt = indicators::option;

static constexpr auto colors = std::array<indicators::Color, 8>{
    indicators::Color::red,     indicators::Color::green,
    indicators::Color::yellow,  indicators::Color::blue,
    indicators::Color::magenta, indicators::Color::cyan,
    indicators::Color::white,   indicators::Color::grey,
};

namespace AZ {

export template <Concepts::Game Game, Concepts::Model Model>
class AlphaZero {
  using State = Game::State;

 public:
  AlphaZero(Config config, std::mt19937& gen) : config_(config), gen_(gen) {
    bars_.set_option(opt::HideBarWhenComplete{true});
  }

  auto learn(Model::Config model_config) -> std::shared_ptr<Model> {
    auto arena = Arena<Game, Model>(config_);

    auto model = std::make_shared<Model>(model_config);
    auto best_model = std::make_shared<Model>(model_config);

    auto optimizer = std::make_shared<torch::optim::Adam>(
        model->parameters(), torch::optim::AdamOptions(0.001));

    for (auto i : std::views::iota(0, config_.num_iterations)) {
      auto memory = Memory{config_, gen_};

      generate_self_play_data(memory, model);

      train(memory, model, optimizer);
      auto [wins, draws, losses] = arena.play(
          model, best_model, config_.num_model_evaluation_iterations,
          /*num_simulations=*/config_.num_model_evaluation_simulations);

      auto did_win =
          wins + draws >
          0.7 * static_cast<float32_t>(config_.num_model_evaluation_iterations);

      if (did_win) {
        best_model = clone_model(model, config_.device);
        save_model(model, std::format("models/model_{}.pt", i));
      }

      std::print("[Iteration {}] WDL - {}:{}:{}", i, wins, draws, losses);
    }

    return best_model;
  }

 private:
  auto train(Memory& memory, std::shared_ptr<Model> model,
             std::shared_ptr<torch::optim::Optimizer> optimizer) -> void {
    namespace F = torch::nn::functional;

    auto bar = std::make_unique<indicators::ProgressBar>(
        opt::BarWidth{50}, opt::ForegroundColor{indicators::Color::white},
        opt::ShowElapsedTime{true}, opt::ShowRemainingTime{true},
        opt::ShowPercentage{true},
        opt::MaxProgress{config_.num_training_epochs},
        opt::PrefixText{"Training Epoch "},
        opt::FontStyles{
            std::vector<indicators::FontStyle>{indicators::FontStyle::bold}});
    auto bar_id = bars_.push_back(std::move(bar));

    if (memory.size() % config_.batch_size == 1)
      memory.pop();

    model->train();
    model->to(config_.device);
    memory.shuffle();

    for (auto _ : std::views::iota(0, config_.num_training_epochs)) {
      auto epoch_loss = 0.;
      for (size_t i = 0; i < memory.size(); i += config_.batch_size) {
        auto [feature, target_value, target_policy] = memory.sample_batch(i);
        auto [out_value, out_policy] = model->forward(feature);

        auto loss = F::cross_entropy(out_value, target_value) +
                    F::cross_entropy(out_policy, target_policy);

        optimizer->zero_grad();
        loss.backward();
        optimizer->step();

        epoch_loss += loss.template item<float32_t>();
      }

      bars_[bar_id].tick();
    }

    bars_[bar_id].mark_as_completed();
  }

  auto run_actor(Memory& memory, std::shared_ptr<Model> model, int32_t actor_id)
      -> void {
    auto color = colors[actor_id % config_.num_actors];
    auto bar = std::make_unique<indicators::ProgressBar>(
        opt::BarWidth{50}, opt::ForegroundColor{color},
        opt::ShowElapsedTime{true}, opt::ShowRemainingTime{true},
        opt::ShowPercentage{true},
        opt::MaxProgress{config_.num_self_play_iterations_per_actor},
        opt::PrefixText{std::format("Thread {} ", actor_id)},
        opt::FontStyles{
            std::vector<indicators::FontStyle>{indicators::FontStyle::bold}});
    auto bar_id = bars_.push_back(std::move(bar));

    auto mcts = MCTS<Game, Model>{config_};

    auto num_iterations = config_.num_self_play_iterations_per_actor;

    // Generate a list of indices that use random playout.
    auto n = static_cast<int32_t>(config_.random_playout_percentage *
                                  num_iterations);
    auto random_playout_indices = torch::randint(num_iterations, {n});

    for (auto i : std::views::iota(0, num_iterations)) {
      auto statistics = std::vector<std::tuple<State, torch::Tensor>>();
      auto state = Game::initial_state();
      while (true) {
        auto is_not_random_playout =
            not torch::isin(i, random_playout_indices).item<bool>();

        // If we're performing random playout we set `num_simulations` to be
        // random on MCTS search.
        auto num_simulations =
            is_not_random_playout
                ? config_.num_simulations
                : torch::randint(1, config_.num_simulations, 1).item<int32_t>();
        auto action_probs = mcts.search(
            state, model, num_simulations,
            is_not_random_playout ? std::make_optional(&gen_) : std::nullopt);

        // If we're using random playout we don't include it in the dataset.
        if (is_not_random_playout) {
          statistics.emplace_back(state, action_probs);
        }

        auto action =
            torch::multinomial(action_probs, 1).template item<Action>();

        auto new_state = Game::apply_action(state, action);
        if (auto outcome = Game::get_outcome(new_state, action)) {
          for (auto& [hist_state, hist_probs] : statistics) {
            auto hist_value = hist_state.player == state.player
                                  ? outcome->as_tensor()
                                  : outcome->flip().as_tensor();
            memory.append(Game::encode_state(hist_state), hist_value,
                          hist_probs);
          }
          break;
        }

        state = std::move(new_state);
      }

      bars_[bar_id].tick();
    }

    bars_[bar_id].mark_as_completed();
  }

  auto generate_self_play_data(Memory& memory, std::shared_ptr<Model> model)
      -> void {
    auto threads = std::vector<std::thread>();

    model->eval();

    for (auto i : std::views::iota(0, config_.num_actors)) {
      threads.emplace_back(
          [this, &memory, model, i] { run_actor(memory, model, i); });
    }

    for (auto& thread : threads) {
      thread.join();
    }
  }

 private:
  indicators::DynamicProgress<indicators::ProgressBar> bars_;
  Config config_;
  std::mt19937& gen_;
};

}  // namespace AZ
