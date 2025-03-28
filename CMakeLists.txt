cmake_minimum_required(VERSION 3.31)

set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD 0e5b6991-d74f-4b3d-a41c-cf096e0b2508)
set(CMAKE_CXX_MODULE_STD ON)

project(DamathZero LANGUAGES CXX)

find_package(GTest REQUIRED)
find_package(Torch REQUIRED)

# include(cmake/indicators.cmake)

add_library(DamathZero)
target_sources(DamathZero PUBLIC FILE_SET CXX_MODULES FILES
  src/config.cpp
  src/alphazero.cpp
  src/game.cpp
  src/memory.cpp
  src/mcts.cpp
  src/node.cpp
  src/storage.cpp
)
target_compile_options(DamathZero PUBLIC -Wall -Wextra -Werror -Wpedantic)
target_compile_features(DamathZero PUBLIC cxx_std_23)
target_link_libraries(DamathZero PUBLIC torch) # indicators::indicators

set(_availableGames tictactoe)
set(game tictactoe CACHE STRING "Game to compile: ${_availableGames}")
set_property(CACHE game PROPERTY STRINGS ${_availableGames})

if(${game} IN_LIST _availableGames)
    add_executable(${game} "src/games/${game}.cpp")
    target_link_libraries(${game} PRIVATE DamathZero)
else()
    message(FATAL_ERROR "game must be one of ${_availableGames}")
endif()

add_custom_command(
  OUTPUT ${PROJECT_BINARY_DIR}/thesis.pdf
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/docs
  COMMAND TEXMF_OUTPUT_DIRECTORY=${PROJECT_BINARY_DIR} latexmk -pdf -bibtex -shell-escape --output-directory=${PROJECT_BINARY_DIR} --aux-directory=${PROJECT_BINARY_DIR} thesis.tex
)

add_custom_target(DamathZeroPaper ALL DEPENDS ${PROJECT_BINARY_DIR}/thesis.pdf)
