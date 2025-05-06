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
    size_t batch_size = 512;

    int32_t num_iterations = 100;
    int32_t num_training_epochs = 10;

    int32_t num_self_play_games = 512;
    int32_t num_self_play_simulations = 100;

    int32_t num_evaluation_games = 64;
    int32_t num_evaluation_simulations = 1000;

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

    auto optimizer = std::make_shared<torch::optim::AdamW>(model->parameters());

    for (auto i : std::views::iota(0, config_.num_iterations)) {
      auto bar = std::make_unique<indicators::ProgressBar>(
          opt::BarWidth{50}, opt::ForegroundColor{colors[i % 6]},
          opt::ShowElapsedTime{true}, opt::ShowRemainingTime{true},
          opt::ShowPercentage{true},
          opt::MaxProgress{config_.num_self_play_games +
                           config_.num_training_epochs +
                           config_.num_evaluation_games},
          opt::PrefixText{
              std::format("Iteration {}/{} ", i + 1, config_.num_iterations)},
          opt::FontStyles{
              std::vector<indicators::FontStyle>{indicators::FontStyle::bold}});
      auto bar_id = bars_.push_back(std::move(bar));

      bars_[bar_id].set_option(opt::PostfixText{"Generating Self-Play Data"});
      auto memory = generate_self_play_data(best_model, bar_id);

      bars_[bar_id].set_option(opt::PostfixText{"Training Model"});
      auto average_loss = train(memory, model, optimizer, bar_id);

      bars_[bar_id].set_option(opt::PostfixText{"Evaluating Model"});
      auto [wins, draws, losses] = evaluate(model, best_model, bar_id);

      auto did_win = wins + draws >
                     0.7 * static_cast<float32_t>(config_.num_evaluation_games);

      if (did_win) {
        best_model = utils::clone_model(model);
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
  auto generate_self_play_data(std::shared_ptr<Model> model, int32_t bar_id)
      -> Memory {
    model->eval();
    using History = std::vector<std::tuple<State, torch::Tensor>>;

    auto mcts = MCTS<Game, Model>{{}};
    auto memory = Memory(gen_);
    auto histories = std::vector<History>(config_.num_self_play_games, History{});

    auto on_game_end = [this, &histories, &memory, &bar_id](size_t game_index, GameOutcome outcome,
                                      Player terminal_player) {
      for (const auto& [hist_state, hist_probs] : histories[game_index]) {
        auto hist_value = hist_state.player == terminal_player
                              ? outcome.as_tensor()
                              : outcome.flip().as_tensor();
        memory.append(Game::encode_state(hist_state), hist_value, hist_probs);
      }

      bars_[bar_id].tick();
    };

    auto on_game_move = [&histories](int32_t game_index, State state,
                               torch::Tensor action_probs) {
      histories[game_index].emplace_back(state, action_probs);
    };

    auto parallel_games = ParallelGames<Game>(config_.num_self_play_games, on_game_end, on_game_move);

    while (not parallel_games.all_terminated()) {
      auto states = parallel_games.get_non_terminal_states();
      auto action_probs =
          mcts.search(states, model, config_.num_self_play_simulations);
      parallel_games.apply_to_non_terminal_states(action_probs);
    }

    return memory;
  }

  auto train(Memory& memory, std::shared_ptr<Model> model,
             std::shared_ptr<torch::optim::Optimizer> optimizer, int32_t bar_id)
      -> float32_t {
    if (memory.size() % config_.batch_size == 1)
      memory.pop();

    model->train();
    memory.shuffle();

    auto device = model->parameters().begin()->device();

    auto total_loss = 0.;
    for (auto i : std::views::iota(0, config_.num_training_epochs)) {
      auto epoch_loss = 0.;
      for (size_t start_index = 0;
           start_index + config_.batch_size < memory.size();
           start_index += config_.batch_size) {
        auto [feature, target_value, target_policy] =
            memory.sample_batch(config_.batch_size, start_index);
        auto [out_value, out_policy] = model->forward(feature.to(device));

        auto loss = F::cross_entropy(out_value, target_value) +
                    F::cross_entropy(out_policy, target_policy);

        optimizer->zero_grad();
        loss.backward();
        optimizer->step();

        epoch_loss += loss.template item<double>();

        bars_[bar_id].set_option(opt::PostfixText{
            std::format("Training Model | Epoch: {}/{} - Batch Loss: {:.6f} - "
                        "Total Loss: {:.6f}",
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

    auto mcts = MCTS<Game, Model>{{}};

    auto memory = Memory(gen_);

    int32_t wins = 0;
    int32_t draws = 0;
    int32_t losses = 0;

    auto on_game_end = [this, &wins, &draws, &losses, &bar_id](size_t, GameOutcome outcome, Player terminal_player) {
      // outcome from the perspective of the current model
      outcome = terminal_player.is_first() ? outcome : outcome.flip();
      if (outcome == GameOutcome::Win) {
        wins++;
      } else if (outcome == GameOutcome::Draw) {
        draws++;
      } else {
        losses++;
      }

      bars_[bar_id].tick();
    };


    auto parallel_games = ParallelGames<Game>(config_.num_evaluation_games, on_game_end);

    while (not parallel_games.all_terminated()) {
      const auto states = parallel_games.get_non_terminal_states();

      const auto action_probs_of_the_current_model = mcts.search(states, current_model, config_.num_self_play_simulations);
      const auto action_probs_of_the_best_model = mcts.search(states, best_model, config_.num_self_play_simulations);

      auto action_probs = torch::zeros({static_cast<int32_t>(states.size()), Game::ActionSize}, torch::kFloat);
      for (const auto i : std::views::iota(0, static_cast<int32_t>(states.size()))) {
          if (states[i].player.is_first()) {
              action_probs[i] = action_probs_of_the_current_model[i];
          } else {
              action_probs[i] = action_probs_of_the_best_model[i];
          }
      }
      parallel_games.apply_to_non_terminal_states(action_probs);
    }

    return {wins, draws, losses};
  }

 private:
  indicators::DynamicProgress<indicators::ProgressBar> bars_;
  Config config_;
  std::mt19937 gen_;
};

}  // namespace az
