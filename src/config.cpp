export module damathzero:config;

import std;

namespace DamathZero {

export struct Config {
  int NumSelfPlayIterations = 500;
  int NumSimulations = 60;
  int NumIterations = 3;
  int NumTrainingEpochs = 4;

  std::size_t BatchSize = 64;

  double C = 2.0;
};

}  // namespace DamathZero
