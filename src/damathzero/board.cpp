
#include "damathzero/board.hpp"

#include <array>
#include <cassert>
#include <vector>

#include "alphazero/game.hpp"

namespace dz {

auto Board::get_jump_actions(int8_t x, int8_t y) const
    -> std::vector<az::Action> {
  assert((x >= 0 and x < 8));
  assert((y >= 0 and y < 8));

  const auto piece = cells[y][x];
  assert(piece.is_occupied);

  auto actions = std::vector<az::Action>{};
  for (auto direction = 0; direction < 4; direction++) {
    const auto [dx, dy] = directions[direction];

    if (not piece.is_knighted) {
      if (piece.is_owned_by_first_player and dy == -1)
        continue;

      if (not piece.is_owned_by_first_player and dy == 1)
        continue;
    }

    for (auto distance = 1; validate(x + distance * dx, y + distance * dy);
         distance++) {
      if (not piece.is_knighted and distance > 1)
        break;

      if (cells[y + distance * dy][x + distance * dx].is_occupied)
        break;

      auto action =
          (8 * 8 * 4 * (distance - 1)) + (8 * 8 * direction) + (8 * y) + x;
      actions.push_back(action);
    }
  }

  return actions;
}

auto Board::get_eatable_actions(int8_t x, int8_t y) const
    -> std::vector<az::Action> {
  assert((x >= 0 and x < 8));
  assert((y >= 0 and y < 8));

  const auto piece = cells[y][x];
  assert(piece.is_occupied);

  auto actions = std::vector<az::Action>{};
  for (auto direction = 0; direction < 4; direction++) {
    const auto [dx, dy] = directions[direction];
    for (auto distance = 1, enemy_seen = 0;
         validate(x + distance * dx, y + distance * dy); distance++) {
      if (not piece.is_knighted and distance > 2)
        break;

      if (enemy_seen > 1)
        break;

      auto cell = cells[y + distance * dy][x + distance * dx];
      if (cell.is_occupied) {
        if (cell.has_same_owner(piece))
          break;
        enemy_seen++;
        continue;
      }

      if (enemy_seen == 1) {
        auto action =
            (8 * 8 * 4 * (distance - 1)) + (8 * 8 * direction) + (8 * y) + x;
        actions.push_back(action);
      }
    }
  }

  return actions;
}

}  // namespace dz
