export module damathzero:game;

import std;

namespace DamathZero {

export using Player = int;

export using Action = int;

export using Board = std::vector<int>;

export struct Game {
  static constexpr auto initial_board() -> Board {
    return std::vector<int>(9, 0);
  }

  static constexpr auto apply_action(const Board& board, Action action,
                                     Player player)
      -> std::tuple<Board, Player> {
    auto new_board = board;
    new_board[action] = player;
    return {new_board, -player};
  }

  static constexpr auto legal_actions(const Board& board)
      -> std::vector<Action> {
    std::vector<Action> legal_actions;

    for (std::size_t i = 0; i < board.size(); i++)
      if (board[i] == 0)
        legal_actions.push_back(i);

    return legal_actions;
  }

  static constexpr auto check_win(const Board& board, Action action) -> bool {
    if (action < 0)
      return false;

    auto player = board[action];
    return (board[0] == player and board[1] == player and board[2] == player) or
           (board[3] == player and board[4] == player and board[5] == player) or
           (board[6] == player and board[7] == player and board[8] == player) or
           (board[0] == player and board[3] == player and board[6] == player) or
           (board[1] == player and board[4] == player and board[7] == player) or
           (board[2] == player and board[5] == player and board[8] == player) or
           (board[0] == player and board[4] == player and board[8] == player) or
           (board[2] == player and board[4] == player and board[6] == player);
  }

  static constexpr auto get_value_and_terminated(const Board& board,
                                                 Action action)
      -> std::tuple<double, bool> {
    if (check_win(board, action))
      return {1.0, true};
    else if (legal_actions(board).empty())
      return {0.0, true};
    else
      return {0.0, false};
  }

  static constexpr auto get_opponent(Player player) -> Player {
    return -player;
  }

  static constexpr auto get_opponent_value(double value) -> double {
    return -value;
  }

  static constexpr auto change_perspective(const Board& board, Player player)
      -> Board {
    Board new_board = board;
    for (auto& cell : new_board)
      cell *= player;
    return new_board;
  }
};

}  // namespace DamathZero
