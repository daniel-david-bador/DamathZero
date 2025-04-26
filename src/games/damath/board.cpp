module;

#include <cassert>

export module dz:board;

import az;
import std;

namespace dz {

export struct Board {
  struct Cell {
    constexpr auto get_value() const -> float {
      assert(is_occupied);
      return is_negative ? -value : value;
    }

    constexpr auto is_owned_by(az::Player player) const -> bool {
      assert(is_occupied);
      return player.is_first() == is_owned_by_first_player;
    }

    constexpr auto has_same_owner(Cell other) const -> bool {
      assert(is_occupied);
      assert(other.is_occupied);
      return is_owned_by_first_player == other.is_owned_by_first_player;
    }

    uint8_t is_occupied : 1;
    uint8_t is_owned_by_first_player : 1;
    uint8_t is_knighted : 1;
    uint8_t is_negative : 1;
    uint8_t value : 4;
  };

  static constexpr auto EmptyCell = Cell{0, 0, 0, 0, 0};

  static constexpr std::array<std::pair<int, int>, 4> directions{
      {{-1, 1}, {1, 1}, {-1, -1}, {1, -1}}};

  static constexpr std::array<std::array<const char, 8>, 8> operators{
      {{' ', '+', ' ', '-', ' ', '/', ' ', '*'},
       {'-', ' ', '+', ' ', '*', ' ', '/', ' '},
       {' ', '/', ' ', '*', ' ', '+', ' ', '-'},
       {'*', ' ', '/', ' ', '-', ' ', '+', ' '},
       {' ', '+', ' ', '-', ' ', '/', ' ', '*'},
       {'-', ' ', '+', ' ', '*', ' ', '/', ' '},
       {' ', '/', ' ', '*', ' ', '+', ' ', '-'},
       {'*', ' ', '/', ' ', '-', ' ', '+', ' '}}};

  // struct Direction {
  //   int8_t x;
  //   int8_t y;
  // };

  // TODO:
  // Use the same concept above to create a constexpr constructor and create a
  // board easily.
  std::array<std::array<Cell, 8>, 8> cells{{
      // clang-format off
        {{{0,0,0,0,0},{1,1,0,1,11},{0,0,0,0,0},{1,1,0,0,8},{0,0,0,0,0},{1,1,0,1,5},{0,0,0,0,0},{1,1,0,0,2}}},
        {{{1,1,0,0,0},{0,0,0,0,0},{1,1,0,1,3},{0,0,0,0,0},{1,1,0,0,10},{0,0,0,0,0},{1,1,0,1,7},{0,0,0,0,0}}},
        {{{0,0,0,0,0},{1,1,0,1,9},{0,0,0,0,0},{1,1,0,0,6},{0,0,0,0,0},{1,1,0,1,1},{0,0,0,0,0},{1,1,0,0,4}}},
        {{{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}}},
        {{{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}}},
        {{{1,0,0,0,4},{0,0,0,0,0},{1,0,0,1,1},{0,0,0,0,0},{1,0,0,0,6},{0,0,0,0,0},{1,0,0,1,9},{0,0,0,0,0}}},
        {{{0,0,0,0,0},{1,0,0,1,7},{0,0,0,0,0},{1,0,0,0,10},{0,0,0,0,0},{1,0,0,1,3},{0,0,0,0,0},{1,0,0,0,0}}},
        {{{1,0,0,0,2},{0,0,0,0,0},{1,0,0,1,5},{0,0,0,0,0},{1,0,0,0,8},{0,0,0,0,0},{1,0,0,1,11},{0,0,0,0,0}}},
      }};  // clang-format on

  constexpr auto operator[](int8_t x, int8_t y) const -> Cell {
    return cells[y][x];
  }
  constexpr auto operator[](int8_t x, int8_t y) -> Cell& { return cells[y][x]; }

  static constexpr auto validate(int x, int y) -> bool {
    return x >= 0 and x < 8 and y >= 0 and y < 8;
  };

  constexpr auto flip() const -> Board {
    return {
        .cells =  // clang-format off
              {{{{cells[7][7], cells[7][6], cells[7][5], cells[7][4], cells[7][3], cells[7][2], cells[7][1], cells[7][0]}},
                {{cells[6][7], cells[6][6], cells[6][5], cells[6][4], cells[6][3], cells[6][2], cells[6][1], cells[6][0]}},
                {{cells[5][7], cells[5][6], cells[5][5], cells[5][4], cells[5][3], cells[5][2], cells[5][1], cells[5][0]}},
                {{cells[4][7], cells[4][6], cells[4][5], cells[4][4], cells[4][3], cells[4][2], cells[4][1], cells[4][0]}},
                {{cells[3][7], cells[3][6], cells[3][5], cells[3][4], cells[3][3], cells[3][2], cells[3][1], cells[3][0]}},
                {{cells[2][7], cells[2][6], cells[2][5], cells[2][4], cells[2][3], cells[2][2], cells[2][1], cells[2][0]}},
                {{cells[1][7], cells[1][6], cells[1][5], cells[1][4], cells[1][3], cells[1][2], cells[1][1], cells[1][0]}},
                {{cells[0][7], cells[0][6], cells[0][5], cells[0][4], cells[0][3], cells[0][2], cells[0][1], cells[0][0]}}
              }},  // clang-format on
    };
  }

  auto get_jump_actions(int x, int y) const -> std::vector<az::Action> {
    assert((x >= 0 and x < 8));
    assert((y >= 0 and y < 8));

    const auto piece = cells[y][x];
    assert(piece.is_occupied);

    auto actions = std::vector<az::Action>{};
    for (auto direction = 0; direction < 4; direction++) {
      const auto [dx, dy] = directions[direction];
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

  auto get_eatable_actions(int32_t x, int32_t y) const
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
};

}  // namespace dz
