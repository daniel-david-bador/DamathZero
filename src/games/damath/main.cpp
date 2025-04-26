#include <raylib.h>

import dz;
import std;

auto Update(dz::Application&) -> void;
auto Render(const dz::Application&) -> void;

auto main(int, char*[]) -> int {
  auto app = dz::Application{};

  InitWindow(1300, 800, "DamathZero");
  SetTargetFPS(60);

  while (not WindowShouldClose()) {
    Update(app);
    Render(app);
  }

  return 0;
}

auto Update(dz::Application& app) -> void {
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

    if (mousePosition.x < 1300 and mousePosition.x > 925 and
        mousePosition.y < 800 and mousePosition.y > 650) {
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

      if (app.moveable_pieces[i][j])
        DrawRectangle(i * 100, (7 - j) * 100, 100, 100, Fade(YELLOW, 0.25));

      if (app.destinations[i][j]) {
        DrawRectangle(i * 100, (7 - j) * 100, 100, 100, Fade(YELLOW, 0.25));
      }

      auto cell = app.state.board[i, j];
      if (cell.is_occupied) {
        DrawCircle(i * 100 + 50, (7 - j) * 100 + 50, 25,
                   cell.is_owned_by(app.state.player) ? MAROON : BLACK);
        DrawTextCenter(GetFontDefault(), std::format("{}", cell.get_value()),
                       i * 100, (7 - j) * 100, 100, 100, 20, 3, WHITE);
      } else {
        DrawTextCenter(
            GetFontDefault(),
            app.destinations[i][j]
                ? std::format(
                      "{:.2f}",
                      app.predicted_action_probs
                          [app.action_map[app.selected_piece.value().first]
                                         [app.selected_piece.value().second][i]
                                         [j]
                                             .value()]
                              .item<float>())
                : std::format("{}", app.state.board.operators[j][i]),
            i * 100, (7 - j) * 100, 100, 100, 20, 3, BLACK);
      }
    }
  }

  auto predicted_win = app.predicted_wdl[0].item<float>() * 800;
  auto predicted_draw = app.predicted_wdl[1].item<float>() * 800;
  auto predicted_loss = app.predicted_wdl[2].item<float>() * 800;

  DrawLineEx({800, 0}, {800, predicted_loss * 800}, 30, BLUE);
  DrawLineEx({800, predicted_loss}, {800, predicted_loss + predicted_draw}, 30,
             GRAY);
  DrawLineEx({800, predicted_loss + predicted_draw},
             {800, predicted_loss + predicted_draw + predicted_win}, 30, GREEN);

  DrawRectangle(800, 0, 500, 800, MAROON);
  DrawTextCenter(GetFontDefault(), "DamathZero", 800, 0, 500, 100, 40, 3,
                 WHITE);
  DrawTextCenter(GetFontDefault(), "Scores", 800, 100, 500, 100, 40, 3, WHITE);
  DrawTextCenter(GetFontDefault(),
                 std::format("Player 1: {:7.2f}", app.state.scores.first), 800,
                 200, 250, 100, 20, 3, WHITE);
  DrawTextCenter(GetFontDefault(),
                 std::format("Player 2: {:7.2f}", app.state.scores.second),
                 1050, 200, 250, 100, 20, 3, WHITE);

  DrawRectangle(925, 650, 250, 50, GREEN);
  DrawTextCenter(GetFontDefault(),
                 std::format("Reset Game", app.state.scores.second), 925, 650,
                 250, 50, 40, 3, WHITE);

  EndDrawing();
}
