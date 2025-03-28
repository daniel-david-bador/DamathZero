export module damathzero:config;

import std;

namespace DamathZero {

export struct Config {
  int NumIterations = 10;
  int NumSimulations = 100;

  int NumSelfPlayIterations = 1000;
  int NumTrainingEpochs = 4;
  int NumModelEvaluationIterations = 50;

  std::size_t BatchSize = 64;

  double C = 2.0;

  double DirichletAlpha = 0.3;
  double DirichletEpsilon = 0.25;
};

}  // namespace DamathZero
