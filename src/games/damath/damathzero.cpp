export module dz;

export import :model;
export import :game;
export import :agent;
export import :board;

import az;

namespace dz {

export using Action = az::Action;
export using Config = az::Config;
export using Player = az::Player;
export using GameOutcome = az::GameOutcome;

export using DamathZero = az::AlphaZero<dz::Game, dz::Model>;

};  // namespace dz
