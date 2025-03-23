cmake_minimum_required(VERSION 3.31)
project(DamathZero LANGUAGES CXX)

find_package(GTest REQUIRED)
find_package(Torch REQUIRED)

add_library(DamathZero)
target_sources(DamathZero PUBLIC FILE_SET CXX_MODULES FILES
  src/config.cpp
  src/mcts.cpp
  src/game.cpp  
  src/network.cpp
  src/damathzero.cpp)
target_compile_options(DamathZero PUBLIC -Wall -Wextra -Werror -Wpedantic)
target_compile_features(DamathZero PUBLIC cxx_std_23)
target_link_libraries(DamathZero PUBLIC torch)

enable_testing()

add_executable(DamathZeroTest)
target_sources(DamathZeroTest PRIVATE test/test_game.cpp test/test_mcts.cpp)
target_link_libraries(DamathZeroTest PRIVATE DamathZero GTest::gtest GTest::gtest_main)

gtest_discover_tests(DamathZeroTest)