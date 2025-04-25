#include <raylib.h>
#include <torch/torch.h>

import dz;
import std;

auto Update(dz::Game::State&) -> void;
auto Render(const dz::Game::State&) -> void;

auto main(int, char*[]) -> int {
  auto damathzero = dz::DamathZero{dz::Config{
      .num_iterations = 1,
      .num_simulations = 10,
      .num_self_play_iterations_per_actor = 10,
      .num_actors = 5,
      .num_model_evaluation_iterations = 5,
      .num_model_evaluation_simulations = 100,
      .device = torch::kCPU,
  }};

  auto model = std::make_shared<dz::Model>(dz::Model::Config{
      .action_size = dz::Game::ActionSize,
      .num_blocks = 2,
      .num_attention_head = 4,
      .embedding_dim = 64,
      .mlp_hidden_size = 128,
      .mlp_dropout_prob = 0.1,
  });

  auto state = dz::Game::initial_state();
  std::cout << dz::Game::legal_actions(state).nonzero();

  InitWindow(1300, 800, "DamathZero");
  SetTargetFPS(60);

  while (not WindowShouldClose()) {
    Update(state);
    Render(state);
  }

  return 0;
}

auto Update(dz::Game::State&) -> void {}

auto Render(const dz::Game::State& state) -> void {
  BeginDrawing();

  ClearBackground(BLACK);

  DrawRectangle(800, 0, 500, 800, MAROON);

  for (auto i = 0; i < 8; i++) {
    for (auto j = 0; j < 8; j++) {
      if (i % 2 != j % 2)
        DrawRectangle(i * 100, (7 - j) * 100, 100, 100, WHITE);

      auto cell = state.board[i, j];
      if (cell.occup) {
        DrawCircle(i * 100 + 50, (7 - j) * 100 + 50, 25,
                   cell.enemy ? BLACK : MAROON);
        DrawText(std::to_string(static_cast<int32_t>(cell.get_value())).c_str(),
                 i * 100 + 40, (7 - j) * 100 + 30, 20, WHITE);
      } else {
        DrawText(std::format("{}", state.board.operators[j][i]).c_str(),
                 i * 100 + 40, (7 - j) * 100 + 40, 20, BLACK);
      }
    }
  }

  EndDrawing();
}
