cmake_minimum_required(VERSION 4.0)
project(DamathZero LANGUAGES CXX)

find_package(Torch REQUIRED)
find_package(raylib REQUIRED)

include(FetchContent)
include(cmake/indicators.cmake)

add_library(AlphaZero SHARED)
target_sources(AlphaZero PRIVATE
  src/alphazero/memory.cpp)
target_include_directories(AlphaZero PUBLIC src)
target_compile_features(AlphaZero PUBLIC cxx_std_23)
target_link_libraries(AlphaZero PUBLIC indicators torch)
target_compile_options(AlphaZero PUBLIC -Wall -Wextra -Werror -Wpedantic)

add_library(DamathZero SHARED)
target_sources(DamathZero PRIVATE
  src/damathzero/dz.cpp
  src/damathzero/game.cpp
  src/damathzero/model.cpp
  src/damathzero/board.cpp)
target_include_directories(DamathZero PUBLIC src)
target_link_libraries(DamathZero PUBLIC AlphaZero)
target_compile_features(DamathZero PUBLIC cxx_std_23)

# add_executable(DamathZeroApp "src/app.cpp")
# target_link_libraries(DamathZeroApp PRIVATE raylib DamathZero)

add_executable(DamathZeroTrainer "src/train.cpp")
target_link_libraries(DamathZeroTrainer PRIVATE DamathZero)

# add_custom_command(
#   OUTPUT ${PROJECT_BINARY_DIR}/thesis.pdf
#   WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/docs
#   COMMAND TEXMF_OUTPUT_DIRECTORY=${PROJECT_BINARY_DIR} latexmk -pdf -bibtex -shell-escape --output-directory=${PROJECT_BINARY_DIR} --aux-directory=${PROJECT_BINARY_DIR} thesis.tex)
# add_custom_target(DamathZeroPaper ALL DEPENDS ${PROJECT_BINARY_DIR}/thesis.pdf)
