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
  }
}

auto DrawTextCenter(Font font, const char* text, int x, int y, int width,
                    int height, float fontSize, float spacing, Color tint)
    -> void {
  auto [textWidth, textHeight] = MeasureTextEx(font, text, fontSize, 0);
  auto textX = x + (width - textWidth) / 2;
  auto textY = y + (height - textHeight) / 2;
  DrawTextEx(font, text, Vector2{textX, textY}, fontSize, spacing, tint);
}

auto Render(const dz::Application& app) -> void {
  BeginDrawing();

  ClearBackground(BLACK);

  for (auto i = 0; i < 8; i++) {
    for (auto j = 0; j < 8; j++) {
      if (i % 2 != j % 2)
        DrawRectangle(i * 100, (7 - j) * 100, 100, 100, WHITE);

      if (app.moveable_pieces[i][j] or app.destinations[i][j])
        DrawRectangle(i * 100, (7 - j) * 100, 100, 100, Fade(YELLOW, 0.25));

      auto cell = app.state.board[i, j];
      if (cell.occup) {
        DrawCircle(i * 100 + 50, (7 - j) * 100 + 50, 25,
                   cell.enemy ? BLACK : MAROON);
        DrawTextCenter(GetFontDefault(),
                       std::format("{}", cell.get_value()).c_str(), i * 100,
                       (7 - j) * 100, 100, 100, 20, 0, WHITE);
      } else {
        DrawTextCenter(
            GetFontDefault(),
            std::format("{}", app.state.board.operators[j][i]).c_str(), i * 100,
            (7 - j) * 100, 100, 100, 20, 0, BLACK);
      }
    }
  }

  DrawRectangle(800, 0, 500, 800, MAROON);
  DrawTextCenter(GetFontDefault(), "DamathZero", 800, 0, 500, 100, 40, 1,
                 WHITE);

  EndDrawing();
}
