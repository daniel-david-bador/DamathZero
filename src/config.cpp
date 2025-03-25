export module damathzero:config;

namespace DamathZero {

export struct Config {
  static constexpr int NumIterations = 1000;
  static constexpr int NumSelfPlayIterations = 1000;
  static constexpr int NumTrainingEpochs = 1000;
  static constexpr int NumSimulations = 1000;
  static constexpr int BatchSize = 64;

  static constexpr double C = 2.0;
};

}  // namespace DamathZero
