#include <raylib.h>

import dz;
import std;

auto Update(dz::Application&) -> void;
auto Render(const dz::Application&) -> void;

auto main(int argc, char* argv[]) -> int {
  if (argc < 2) {
    std::println(std::cerr, "Expected model path as the first argument.");
    return -1;
  }

  auto app = dz::Application{dz::Config{
                                 .num_iterations = 1,
                                 .num_simulations = 1000,
                                 .num_self_play_iterations_per_actor = 10,
                                 .num_actors = 5,
                                 .num_model_evaluation_iterations = 5,
                                 .num_model_evaluation_simulations = 100,
                                 .device = dz::DeviceType::CPU,
                             },
                             dz::Model::Config{
                                 .action_size = dz::Game::ActionSize,
                                 .num_blocks = 10,
                                 .num_attention_head = 4,
                                 .embedding_dim = 64,
                                 .mlp_hidden_size = 128,
                                 .mlp_dropout_prob = 0.1,
                             },
                             argv[1]};

  InitWindow(1330, 830, "DamathZero");
  SetTargetFPS(60);

  while (not WindowShouldClose()) {
    Update(app);
    Render(app);
  }

  return 0;
}

auto Update(dz::Application& app) -> void {
  if (app.state.player.is_second() and not app.outcome.has_value())
    app.let_ai_move();

  if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
    auto mousePosition = GetMousePosition();
    if (mousePosition.x < 800 and mousePosition.y < 800) {
      auto x = static_cast<int>(mousePosition.x / 100);
      auto y = static_cast<int>((800 - mousePosition.y) / 100);
      if (app.selected_piece and app.destinations[x][y])
        app.move_piece_to(x, y);
      else if (app.moveable_pieces[x][y])
        app.select_piece(x, y);
      else
        app.unselect_piece();
    }

    if (mousePosition.x > 825 and mousePosition.x < 1025 and
        mousePosition.y > 650 and mousePosition.y < 700) {
      app.undo_move();
    }

    if (mousePosition.x > 1075 and mousePosition.x < 1175 and
        mousePosition.y > 650 and mousePosition.y < 700) {
      app.reset_game();
    }
  }
}

auto DrawTextCenter(Font font, std::string_view text, int x, int y, int width,
                    int height, float fontSize, float spacing, Color tint)
    -> void {
  auto [textWidth, textHeight] =
      MeasureTextEx(font, text.data(), fontSize, spacing);
  auto textX = x + (width - textWidth) / 2;
  auto textY = y + (height - textHeight) / 2;
  DrawTextEx(font, text.data(), Vector2{textX, textY}, fontSize, spacing, tint);
}

auto Render(const dz::Application& app) -> void {
  BeginDrawing();

  ClearBackground(BLACK);

  for (auto i = 0; i < 8; i++) {
    for (auto j = 0; j < 8; j++) {
      if (i % 2 != j % 2)
        DrawRectangle(i * 100, (7 - j) * 100, 100, 100, WHITE);

      if (app.moveable_pieces[i][j]) {
        auto max_action_probs = app.max_action_probs(i, j);
        DrawRectangle(
            i * 100, (7 - j) * 100, 100, 100,
            Fade(ColorLerp(RED, YELLOW, max_action_probs * 0.25), 0.25));
      }

      if (app.destinations[i][j]) {
        auto action_probs = app.action_probs(i, j);
        DrawRectangle(i * 100, (7 - j) * 100, 100, 100,
                      Fade(ColorLerp(RED, YELLOW, action_probs * 0.25), 0.25));
      }

      auto cell = app.state.board[i, j];
      if (cell.is_occupied) {
        if (cell.is_knighted)
          DrawCircle(i * 100 + 50, (7 - j) * 100 + 50, 30, GRAY);

        DrawCircle(i * 100 + 50, (7 - j) * 100 + 50, 25,
                   cell.is_owned_by_first_player ? MAROON : BLACK);
        DrawTextCenter(GetFontDefault(), std::format("{}", cell.value()),
                       i * 100, (7 - j) * 100, 100, 100, 20, 3, WHITE);
      } else {
        DrawTextCenter(GetFontDefault(),
                       std::format("{}", app.state.board.operators[j][i]),
                       i * 100, (7 - j) * 100, 100, 100, 20, 3, BLACK);
      }
    }
  }

  if (auto wdl = app.wdl_probs()) {
    auto [win, draw, loss] = wdl.value();
    win *= 800, draw *= 800, loss *= 800;
    DrawRectangle(800, 0, 30, loss * 800, BLUE);
    DrawRectangle(800, loss, 30, loss + draw, GRAY);
    DrawRectangle(800, loss + draw, 30, loss + draw + win, GREEN);
  } else {
    DrawRectangle(800, 0, 30, 800, GRAY);
  }

  auto draw_counter = (app.state.draw_count / 80.0f) * 800;
  DrawRectangle(0, 800, draw_counter, 30, BLUE);

  DrawRectangle(800, 800, 30, 30, YELLOW);

  DrawRectangle(830, 0, 500, 830, MAROON);
  DrawTextCenter(GetFontDefault(), "DamathZero", 830, 0, 500, 100, 40, 3,
                 WHITE);

  auto [score, ai_score] = app.state.scores;
  DrawTextCenter(GetFontDefault(), "Scores", 830, 100, 500, 100, 40, 3, WHITE);
  DrawTextCenter(GetFontDefault(), std::format("You: {:7.2f}", score), 830, 200,
                 250, 100, 20, 3, WHITE);
  DrawTextCenter(GetFontDefault(), std::format("AI: {:7.2f}", ai_score), 1080,
                 200, 250, 100, 20, 3, WHITE);

  if (app.outcome.has_value()) {
    auto outcome = app.outcome.value();
    std::string_view outcome_text;

    if (outcome == dz::GameOutcome::Win)
      outcome_text = "You Win!";
    else if (outcome == dz::GameOutcome::Draw)
      outcome_text = "Draw";
    else if (outcome == dz::GameOutcome::Loss)
      outcome_text = "You Lose";

    DrawTextCenter(GetFontDefault(), outcome_text, 830, 300, 500, 100, 40, 3,
                   WHITE);
  }

  DrawRectangle(855, 650, 200, 50, GREEN);
  DrawTextCenter(GetFontDefault(), "Undo", 855, 650, 200, 50, 40, 3, WHITE);

  DrawRectangle(1105, 650, 200, 50, GREEN);
  DrawTextCenter(GetFontDefault(), "Reset", 1105, 650, 200, 50, 40, 3, WHITE);

  EndDrawing();
}
