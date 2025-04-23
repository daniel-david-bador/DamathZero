module;

#include <SDL3/SDL.h>

export module damathzero;

export import :model;
export import :game;
export import :agent;

import std;

export struct DamathZero {
  SDL_Window* window;
  SDL_Renderer* renderer;
  std::shared_ptr<Model> model;

  Damath::State state;
};