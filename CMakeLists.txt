cmake_minimum_required(VERSION 4.0)

set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD a9e1cf81-9932-4810-974b-6eccaf14e457)
set(CMAKE_CXX_MODULE_STD ON)

project(AlphaZero LANGUAGES CXX)

find_package(GTest REQUIRED)
find_package(Torch REQUIRED)

include(cmake/indicators.cmake)

add_library(AlphaZero)
target_sources(AlphaZero PUBLIC FILE_SET CXX_MODULES FILES
  src/config.cpp
  src/alphazero.cpp
  src/game.cpp
  src/memory.cpp
  src/mcts.cpp
  src/node.cpp
  src/storage.cpp
  src/arena.cpp
  src/models/transformer.cpp
  src/models/model.cpp
)
target_compile_options(AlphaZero PUBLIC -Wall -Wextra -Werror -Wpedantic)
target_compile_features(AlphaZero PUBLIC cxx_std_23)
target_link_libraries(AlphaZero PRIVATE indicators::indicators) 
target_link_libraries(AlphaZero PUBLIC torch)

add_executable(TicTacToeZero "src/games/tictactoe.cpp")
target_link_libraries(TicTacToeZero PRIVATE AlphaZero)

add_executable(DamathZero "src/games/damath.cpp")
target_link_libraries(DamathZero PRIVATE AlphaZero)

add_custom_command(
  OUTPUT ${PROJECT_BINARY_DIR}/thesis.pdf
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/docs
  COMMAND TEXMF_OUTPUT_DIRECTORY=${PROJECT_BINARY_DIR} latexmk -pdf -bibtex -shell-escape --output-directory=${PROJECT_BINARY_DIR} --aux-directory=${PROJECT_BINARY_DIR} thesis.tex
)

add_custom_target(DamathZeroPaper ALL DEPENDS ${PROJECT_BINARY_DIR}/thesis.pdf)
