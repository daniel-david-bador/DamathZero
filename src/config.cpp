export module damathzero:config;

import std;

namespace DamathZero {

export struct Config {
  static constexpr int NumIterations = 10;
  static constexpr int NumSelfPlayIterations = 10;
  static constexpr int NumTrainingEpochs = 10;
  static constexpr int NumSimulations = 10;

  static constexpr std::size_t BatchSize = 64;

  static constexpr double C = 2.0;
};

}  // namespace DamathZero
