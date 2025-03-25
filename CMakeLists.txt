cmake_minimum_required(VERSION 3.31)

set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD 0e5b6991-d74f-4b3d-a41c-cf096e0b2508)
set(CMAKE_CXX_MODULE_STD ON)

project(DamathZero LANGUAGES CXX)

find_package(GTest REQUIRED)
find_package(Torch REQUIRED)

add_library(DamathZero)
target_sources(DamathZero PUBLIC FILE_SET CXX_MODULES FILES
  src/config.cpp
  src/damathzero.cpp
  src/game.cpp
  src/memory.cpp
  src/mcts.cpp
  src/network.cpp
  src/node.cpp
  src/storage.cpp
)
target_compile_options(DamathZero PUBLIC -Wall -Wextra -Werror -Wpedantic)
target_compile_features(DamathZero PUBLIC cxx_std_23)
target_link_libraries(DamathZero PUBLIC torch)

enable_testing()

add_executable(DamathZeroTest)
target_sources(DamathZeroTest PRIVATE 
  test/test_game.cpp 
  test/test_mcts.cpp
  test/test_damathzero.cpp
)
target_link_libraries(DamathZeroTest PRIVATE DamathZero GTest::gtest GTest::gtest_main)

gtest_discover_tests(DamathZeroTest)
