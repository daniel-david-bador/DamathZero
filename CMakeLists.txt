cmake_minimum_required(VERSION 4.0)

set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD a9e1cf81-9932-4810-974b-6eccaf14e457)
set(CMAKE_CXX_MODULE_STD ON)

project(DamathZero LANGUAGES CXX)

find_package(Torch REQUIRED)
find_package(raylib REQUIRED)

include(FetchContent)
include(cmake/indicators.cmake)

add_library(AlphaZero SHARED)
target_sources(AlphaZero PUBLIC FILE_SET CXX_MODULES FILES
  src/alphazero/az.cpp
  src/alphazero/config.cpp
  src/alphazero/game.cpp
  src/alphazero/memory.cpp
  src/alphazero/mcts.cpp
  src/alphazero/node.cpp
  src/alphazero/storage.cpp
  src/alphazero/model.cpp)
target_compile_options(AlphaZero PUBLIC -Wall -Wextra -Werror -Wpedantic)
target_compile_features(AlphaZero PUBLIC cxx_std_23)
target_link_libraries(AlphaZero PRIVATE indicators::indicators)
target_link_libraries(AlphaZero PUBLIC torch)

add_library(DamathZero SHARED)
target_sources(DamathZero PUBLIC FILE_SET CXX_MODULES FILES
  src/damathzero/dz.cpp
  src/damathzero/game.cpp
  src/damathzero/model.cpp
  src/damathzero/vit.cpp
  src/damathzero/board.cpp)
target_link_libraries(DamathZero PUBLIC AlphaZero)
target_compile_features(DamathZero PUBLIC cxx_std_23)

add_executable(DamathZeroApp "src/app.cpp")
target_link_libraries(DamathZeroApp PRIVATE raylib DamathZero)

add_executable(DamathZeroTrainer "src/train.cpp")
target_link_libraries(DamathZeroTrainer PRIVATE DamathZero)

add_custom_command(
  OUTPUT ${PROJECT_BINARY_DIR}/thesis.pdf
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/docs
  COMMAND TEXMF_OUTPUT_DIRECTORY=${PROJECT_BINARY_DIR} latexmk -pdf -bibtex -shell-escape --output-directory=${PROJECT_BINARY_DIR} --aux-directory=${PROJECT_BINARY_DIR} thesis.tex)
add_custom_target(DamathZeroPaper ALL DEPENDS ${PROJECT_BINARY_DIR}/thesis.pdf)
