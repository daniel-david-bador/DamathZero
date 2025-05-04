#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <indicators/dynamic_progress.hpp>
#include <indicators/progress_bar.hpp>
#include <random>
#include <ranges>

#include "alphazero/game.hpp"
#include "alphazero/mcts.hpp"
#include "alphazero/memory.hpp"
#include "alphazero/model.hpp"

namespace F = torch::nn::functional;
namespace opt = indicators::option;

static constexpr auto colors = std::array<indicators::Color, 8>{
    indicators::Color::red,     indicators::Color::green,
    indicators::Color::yellow,  indicators::Color::blue,
    indicators::Color::magenta, indicators::Color::cyan,
    indicators::Color::white,   indicators::Color::grey,
};

namespace az {

template <concepts::Game Game, concepts::Model Model>
class AlphaZero {
  using State = Game::State;

  struct Config {
    size_t batch_size = 64;

    int32_t num_training_epochs = 4;
    int32_t num_training_iterations = 10;

    int32_t num_self_play_actors = 6;
    int32_t num_self_play_iterations = 100;
    int32_t num_self_play_simulations = 60;

    int32_t num_evaluation_actors = 5;
    int32_t num_evaluation_iterations = 10;
    int32_t num_evaluation_simulations = 1000;

    float32_t random_playout_percentage = 0.2;

    torch::DeviceType device;
  };

 public:
  AlphaZero(Config config,
            std::mt19937 gen = std::mt19937{std::random_device{}()})
      : config_(std::move(config)), gen_(std::move(gen)) {}

  auto learn(Model::Config model_config,
             std::optional<std::shared_ptr<Model>> previous_model =
                 std::nullopt) -> std::shared_ptr<Model> {
    auto model = previous_model ? *previous_model
                                : std::make_shared<Model>(model_config);
    auto best_model = utils::clone_model<Model>(model);

    model->to(config_.device);
    best_model->to(config_.device);

    auto optimizer = std::make_shared<torch::optim::AdamW>(model->parameters());

    for (auto i : std::views::iota(0, config_.num_training_iterations)) {
      auto bar = std::make_unique<indicators::ProgressBar>(
          opt::BarWidth{50}, opt::ForegroundColor{colors[i % 6]},
          opt::ShowElapsedTime{true}, opt::ShowRemainingTime{true},
          opt::ShowPercentage{true},
          opt::MaxProgress{config_.num_self_play_iterations *
                               config_.num_self_play_actors +
                           config_.num_training_epochs +
                           config_.num_evaluation_iterations *
                               config_.num_evaluation_actors},
          opt::PrefixText{std::format("Iteration {}/{} ", i + 1,
                                      config_.num_training_iterations)},
          opt::FontStyles{
              std::vector<indicators::FontStyle>{indicators::FontStyle::bold}});
      auto bar_id = bars_.push_back(std::move(bar));

      auto memory = Memory{gen_};

      bars_[bar_id].set_option(opt::PostfixText{"Generating Self-Play Data"});
      generate_self_play_data(memory, best_model, bar_id);

      bars_[bar_id].set_option(opt::PostfixText{"Training Model"});
      auto average_loss = train(memory, model, optimizer, bar_id);

      bars_[bar_id].set_option(opt::PostfixText{"Evaluating Model"});
      auto [wins, draws, losses] = evaluate(model, best_model, bar_id);

      auto did_win =
          wins + draws >
          0.7 * static_cast<float32_t>(config_.num_evaluation_iterations);

      if (did_win) {
        best_model = utils::clone_model(model);
        best_model->to(config_.device);
        utils::save_model(model,
                          std::format("models/best_models/model_{}.pt", i));
      }

      utils::save_model(model, std::format("models/all_models/model_{}.pt", i));

      bars_[bar_id].set_option(opt::PostfixText{std::format(
          "Average Loss: {:.6f} - Wins: {} - Draws: {} - Losses: {}",
          average_loss, wins, draws, losses)});

      bars_[bar_id].mark_as_completed();
    }

    return best_model;
  }

 private:
  auto generate_self_play_data(Memory& memory, std::shared_ptr<Model> model,
                               int32_t bar_id) -> void {
    model->eval();

    auto threads = std::vector<std::thread>();
    for (auto _ : std::views::iota(0, config_.num_self_play_actors)) {
      threads.emplace_back([this, &memory, model, bar_id] {
        auto mcts = MCTS<Game, Model>{
            {.num_simulations = config_.num_self_play_simulations}};

        auto num_iterations = config_.num_self_play_iterations;

        // Generate a list of indices that use random playout.
        auto n = static_cast<int32_t>(config_.random_playout_percentage *
                                      num_iterations);
        auto random_playout_indices = torch::randint(num_iterations, {n});

        for (auto i : std::views::iota(0, num_iterations)) {
          auto statistics = std::vector<std::tuple<State, torch::Tensor>>();
          auto state = Game::initial_state();
          while (true) {
            auto is_not_random_playout =
                not torch::isin(i, random_playout_indices)
                        .template item<bool>();

            // If we're performing random playout we set `num_simulations`
            // to be random on MCTS search.
            auto num_simulations =
                is_not_random_playout
                    ? config_.num_self_play_simulations
                    : torch::randint(1, config_.num_self_play_simulations, 1)
                          .template item<int32_t>();
            auto action_probs =
                mcts.search(state, model, num_simulations,
                            is_not_random_playout ? std::make_optional(&gen_)
                                                  : std::nullopt);

            // If we're using random playout we don't include it in the
            // dataset.
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
      });
    }

    for (auto& thread : threads) {
      thread.join();
    }
  }

  auto train(Memory& memory, std::shared_ptr<Model> model,
             std::shared_ptr<torch::optim::Optimizer> optimizer, int32_t bar_id)
      -> float32_t {
    if (memory.size() % config_.batch_size == 1)
      memory.pop();

    model->train();
    memory.shuffle();

    auto total_loss = 0.;
    for (auto i : std::views::iota(0, config_.num_training_epochs)) {
      auto epoch_loss = 0.;
      for (size_t start_index = 0;
           start_index + config_.batch_size < memory.size();
           start_index += config_.batch_size) {
        auto [feature, target_value, target_policy] =
            memory.sample_batch(config_.batch_size, start_index);
        auto [out_value, out_policy] = model->forward(feature);

        auto loss = F::cross_entropy(out_value, target_value) +
                    F::cross_entropy(out_policy, target_policy);

        optimizer->zero_grad();
        loss.backward();
        optimizer->step();

        epoch_loss += loss.template item<double>();

        bars_[bar_id].set_option(opt::PostfixText{
            std::format("Epoch {}/{}: Batch Loss: {:.6f} - Total Loss: {:.6f}",
                        i + 1, config_.num_training_epochs,
                        loss.template item<double>(), epoch_loss)});
        bars_[bar_id].tick();
      }

      total_loss += epoch_loss;
    }

    return total_loss / static_cast<float32_t>(config_.num_training_epochs);
  }

  auto evaluate(std::shared_ptr<Model> current_model,
                std::shared_ptr<Model> best_model, int32_t bar_id)
      -> std::tuple<int32_t, int32_t, int32_t> {
    current_model->eval();
    best_model->eval();

    std::atomic<int32_t> wins{0};
    std::atomic<int32_t> draws{0};
    std::atomic<int32_t> losses{0};

    std::vector<std::thread> threads;

    for (auto _ : std::views::iota(0, config_.num_evaluation_actors)) {
      threads.emplace_back([this, &wins, &draws, &losses, current_model,
                            best_model, &bar_id] {
        for (auto _ : std::views::iota(0, config_.num_evaluation_iterations)) {
          auto state = Game::initial_state();
          auto mcts = MCTS<Game, Model>{
              {.num_simulations = config_.num_evaluation_simulations}};

          while (true) {
            auto model = state.player.is_first() ? current_model : best_model;
            auto action_probs = mcts.search(state, model);

            auto action = torch::argmax(action_probs).template item<Action>();

            auto new_state = Game::apply_action(state, action);

            if (auto outcome = Game::get_outcome(new_state, action)) {
              auto flipped_outcome =
                  state.player.is_first() ? *outcome : outcome->flip();

              if (flipped_outcome == GameOutcome::Win) {
                wins += 1;
              } else if (flipped_outcome == GameOutcome::Draw) {
                draws += 1;
              } else if (flipped_outcome == GameOutcome::Loss) {
                losses += 1;
              }

              bars_[bar_id].set_option(opt::PostfixText{std::format(
                  "Evaluating Model: Wins: {} - Draws: {} - Losses: {}",
                  wins.load(), draws.load(), losses.load())});

              bars_[bar_id].tick();
              break;
            }

            state = std::move(new_state);
          }
        }
      });
    }

    for (auto& thread : threads) {
      thread.join();
    }

    return {wins, draws, losses};
  }

 private:
  indicators::DynamicProgress<indicators::ProgressBar> bars_;
  Config config_;
  std::mt19937 gen_;
};

}  // namespace az
