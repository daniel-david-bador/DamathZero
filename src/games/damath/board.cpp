module;

#include <cassert>

export module dz:board;

import az;
import std;

namespace dz {

export struct Board {
  struct Cell {
    uint8_t occup : 1;
    uint8_t enemy : 1;
    uint8_t queen : 1;
    uint8_t ngtve : 1;
    uint8_t value : 4;

    auto get_value() const -> float {
      assert(occup);
      return ngtve ? -value : value;
    }
  };

  static constexpr auto EmptyCell = Cell{0, 0, 0, 0, 0};

  static constexpr std::array<std::array<const char, 8>, 8> operators{
      {{' ', '+', ' ', '-', ' ', '/', ' ', '*'},
       {'-', ' ', '+', ' ', '*', ' ', '/', ' '},
       {' ', '/', ' ', '*', ' ', '+', ' ', '-'},
       {'*', ' ', '/', ' ', '-', ' ', '+', ' '},
       {' ', '+', ' ', '-', ' ', '/', ' ', '*'},
       {'-', ' ', '+', ' ', '*', ' ', '/', ' '},
       {' ', '/', ' ', '*', ' ', '+', ' ', '-'},
       {'*', ' ', '/', ' ', '-', ' ', '+', ' '}}};

  // TODO:
  // Use the same concept above to create a constexpr constructor and create a
  // board easily.
  std::array<std::array<Cell, 8>, 8> cells{{
      // clang-format off
        {{{0,0,0,0,0},{1,0,0,1,11},{0,0,0,0,0},{1,0,0,0,8},{0,0,0,0,0},{1,0,0,1,5},{0,0,0,0,0},{1,0,0,0,2}}},
        {{{1,0,0,0,0},{0,0,0,0,0},{1,0,0,1,3},{0,0,0,0,0},{1,0,0,0,10},{0,0,0,0,0},{1,0,0,1,7},{0,0,0,0,0}}},
        {{{0,0,0,0,0},{1,0,0,1,9},{0,0,0,0,0},{1,0,0,0,6},{0,0,0,0,0},{1,0,0,1,1},{0,0,0,0,0},{1,0,0,0,4}}},
        {{{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}}},
        {{{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}}},
        {{{1,1,0,0,4},{0,0,0,0,0},{1,1,0,1,1},{0,0,0,0,0},{1,1,0,0,6},{0,0,0,0,0},{1,1,0,1,9},{0,0,0,0,0}}},
        {{{0,0,0,0,0},{1,1,0,1,7},{0,0,0,0,0},{1,1,0,0,10},{0,0,0,0,0},{1,1,0,1,3},{0,0,0,0,0},{1,1,0,0,0}}},
        {{{1,1,0,0,2},{0,0,0,0,0},{1,1,0,1,5},{0,0,0,0,0},{1,1,0,0,8},{0,0,0,0,0},{1,1,0,1,11},{0,0,0,0,0}}},
      }};  // clang-format on

  constexpr auto operator[](int8_t x, int8_t y) const -> Cell {
    return cells[y][x];
  }
  constexpr auto operator[](int8_t x, int8_t y) -> Cell& { return cells[y][x]; }

  constexpr auto flip() const -> Board {
    auto new_board = Board{
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

    for (auto& row : new_board.cells)
      for (auto& piece : row)
        if (piece.occup)
          piece.enemy = not piece.enemy;

    return new_board;
  }

  auto get_jump_actions(int x, int y) const -> std::vector<az::Action> {
    auto actions = std::vector<az::Action>{};

    auto piece = cells[y][x];

    assert(not piece.enemy);

    if (not piece.queen) {
      if (y + 1 < 8) {
        if (x - 1 >= 0 and not cells[y + 1][x - 1].occup) {
          auto action = (8 * 8 * 4 * 0) + (8 * 8 * 0) + (8 * y) + x;
          actions.push_back(action);
        }
        if (x + 1 < 8 and not cells[y + 1][x + 1].occup) {
          auto action = (8 * 8 * 4 * 0) + (8 * 8 * 1) + (8 * y) + x;
          actions.push_back(action);
        }
      }
      return actions;
    }

    for (auto distance = 1; x - distance >= 0 and y + distance < 8; distance++)
      if (not cells[y + distance][x - distance].occup) {
        auto valid = true;

        for (auto between = 1; between < distance; between++)
          if (cells[y + between][x - between].occup)
            valid = false;

        if (valid) {
          auto action =
              (8 * 8 * 4 * (distance - 1)) + (8 * 8 * 0) + (8 * y) + x;
          actions.push_back(action);
        }
      }
    for (auto distance = 1; x + distance < 8 and y + distance < 8; distance++)
      if (not cells[y + distance][x + distance].occup) {
        auto valid = true;

        for (auto between = 1; between < distance; between++)
          if (cells[y + between][x + between].occup)
            valid = false;

        if (valid) {
          auto action =
              (8 * 8 * 4 * (distance - 1)) + (8 * 8 * 1) + (8 * y) + x;
          actions.push_back(action);
        }
      }
    for (auto distance = 1; x - distance >= 0 and y - distance >= 0; distance++)
      if (not cells[y - distance][x - distance].occup) {
        auto valid = true;

        for (auto between = 1; between < distance; between++)
          if (cells[y - between][x - between].occup)
            valid = false;

        if (valid) {
          auto action =
              (8 * 8 * 4 * (distance - 1)) + (8 * 8 * 2) + (8 * y) + x;
          actions.push_back(action);
        }
      }
    for (auto distance = 1; x + distance < 8 and y - distance >= 0; distance++)
      if (not cells[y - distance][x + distance].occup) {
        auto valid = true;

        for (auto between = 1; between < distance; between++)
          if (cells[y - between][x + between].occup)
            valid = false;

        if (valid) {
          auto action =
              (8 * 8 * 4 * (distance - 1)) + (8 * 8 * 3) + (8 * y) + x;
          actions.push_back(action);
        }
      }
    return actions;
  }

  auto get_eatable_actions(int32_t x, int32_t y) const
      -> std::vector<az::Action> {
    auto actions = std::vector<az::Action>{};

    auto cell = cells[y][x];

    if (not cell.occup) {
      return {};
    }

    if (not cell.queen) {
      if (y + 2 < 8) {
        if (x - 2 >= 0 and not cells[y + 2][x - 2].occup and
            cells[y + 1][x - 1].enemy) {
          auto action = (8 * 8 * 4 * 1) + (8 * 8 * 0) + (8 * y) + x;
          actions.push_back(action);
        }
        if (x + 2 < 8 and not cells[y + 2][x + 2].occup and
            cells[y + 1][x + 1].enemy) {
          auto action = (8 * 8 * 4 * 1) + (8 * 8 * 1) + (8 * y) + x;
          actions.push_back(action);
        }
      }
      if (y - 2 >= 0) {
        if (x - 2 >= 0 and not cells[y - 2][x - 2].occup and
            cells[y - 1][x - 1].enemy) {
          auto action = (8 * 8 * 4 * 1) + (8 * 8 * 2) + (8 * y) + x;
          actions.push_back(action);
        }
        if (x + 2 < 8 and not cells[y - 2][x + 2].occup and
            cells[y - 1][x + 1].enemy) {
          auto action = (8 * 8 * 4 * 1) + (8 * 8 * 3) + (8 * y) + x;
          actions.push_back(action);
        }
      }
      return actions;
    }

    for (auto distance = 2; x - distance >= 0 and y + distance < 8;
         distance++) {
      if (cells[y + distance][x - distance].occup)
        continue;

      for (auto enemy = 1; enemy < distance; enemy++) {
        if (not cells[y + enemy][x - enemy].enemy)
          continue;

        auto valid = true;
        for (auto between = enemy + 1; between < distance; between++)
          if (cells[y + between][x - between].occup)
            valid = false;

        if (valid) {
          auto action =
              (8 * 8 * 4 * (distance - 1)) + (8 * 8 * 0) + (8 * y) + x;
          actions.push_back(action);
        }
      }
    }

    for (auto distance = 2; x + distance < 8 and y + distance < 8; distance++) {
      if (cells[y + distance][x + distance].occup)
        continue;

      for (auto enemy = 1; enemy < distance; enemy++) {
        if (not cells[y + enemy][x + enemy].enemy)
          continue;

        auto valid = true;
        for (auto between = enemy + 1; between < distance; between++)
          if (cells[y + between][x + between].occup)
            valid = false;

        if (valid) {
          auto action =
              (8 * 8 * 4 * (distance - 1)) + (8 * 8 * 1) + (8 * y) + x;
          actions.push_back(action);
        }
      }
    }
    for (auto distance = 2; x - distance >= 0 and y - distance >= 0;
         distance++) {
      if (cells[y - distance][x - distance].occup)
        continue;

      for (auto enemy = 1; enemy < distance; enemy++) {
        if (not cells[y - enemy][x - enemy].enemy)
          continue;

        auto valid = true;
        for (auto between = enemy + 1; between < distance; between++)
          if (cells[y - between][x - between].occup)
            valid = false;

        if (valid) {
          auto action =
              (8 * 8 * 4 * (distance - 1)) + (8 * 8 * 2) + (8 * y) + x;
          actions.push_back(action);
        }
      }
    }

    for (auto distance = 2; x + distance < 8 and y - distance >= 0;
         distance++) {
      if (cells[y - distance][x + distance].occup)
        continue;

      for (auto enemy = 1; enemy < distance; enemy++) {
        if (not cells[y - enemy][x + enemy].enemy)
          continue;

        auto valid = true;
        for (auto between = enemy + 1; between < distance; between++)
          if (cells[y - between][x + between].occup)
            valid = false;

        if (valid) {
          auto action =
              (8 * 8 * 4 * (distance - 1)) + (8 * 8 * 3) + (8 * y) + x;
          actions.push_back(action);
        }
      }
    }

    return actions;
  }
};

}  // namespace dz
