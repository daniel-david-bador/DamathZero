#include <torch/torch.h>

#define SDL_MAIN_USE_CALLBACKS
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

import std;
import alphazero;
import damathzero;

SDL_FRect rects[32];
SDL_FRect info;

auto SDL_AppInit(void** appstate, int, char*[]) -> SDL_AppResult {
  *appstate = new DamathZero;

  auto& app = *static_cast<DamathZero*>(*appstate);

  SDL_SetAppMetadata("DamathZero", "1.0", "com.bads.damathzero");

  if (!SDL_Init(SDL_INIT_VIDEO)) {
    SDL_Log("Couldn't initialize SDL: %s", SDL_GetError());
    return SDL_APP_FAILURE;
  }

  if (!SDL_CreateWindowAndRenderer("DamathZero", 1300, 800, 0, &app.window,
                                   &app.renderer)) {
    SDL_Log("Couldn't create window/renderer: %s", SDL_GetError());
    return SDL_APP_FAILURE;
  }

  auto config = AZ::Config{
      .num_iterations = 1,
      .num_simulations = 10,
      .num_self_play_iterations_per_actor = 10,
      .num_actors = 5,
      .num_model_evaluation_iterations = 5,
      .num_model_evaluation_simulations = 100,
      .device = torch::kCPU,
  };
  auto gen = std::mt19937{};

  auto alpha_zero = AZ::AlphaZero<Damath, Model>{
      config,
      gen,
  };

  app.model = alpha_zero.learn({
      .action_size = Damath::ActionSize,
      .num_blocks = 2,
      .num_attention_head = 4,
      .embedding_dim = 64,
      .mlp_hidden_size = 128,
      .mlp_dropout_prob = 0.1,
  });

  int index = 0;
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 8; j++)
      if (i % 2 == j % 2 and (i % 2 == 0 or j % 2 == 1))
        rects[index++] = {i * 100.0f, j * 100.0f, 100, 100};

  info = {800, 0, 500, 800};

  return SDL_APP_CONTINUE;
}

auto SDL_AppEvent(void* appstate, SDL_Event* event) -> SDL_AppResult {
  auto& _ = *static_cast<DamathZero*>(appstate);

  if (event->type == SDL_EVENT_QUIT)
    return SDL_APP_SUCCESS;

  return SDL_APP_CONTINUE;
}

auto SDL_AppIterate(void* appstate) -> SDL_AppResult {
  auto& app = *static_cast<DamathZero*>(appstate);
  auto& renderer = app.renderer;

  SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
  SDL_RenderClear(renderer);

  SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);
  SDL_RenderFillRects(renderer, rects, 32);

  SDL_SetRenderDrawColor(renderer, 128, 0, 0, SDL_ALPHA_OPAQUE);
  SDL_RenderFillRect(renderer, &info);

  SDL_RenderPresent(renderer);

  return SDL_APP_CONTINUE;
}

auto SDL_AppQuit(void* appstate, SDL_AppResult) -> void {
  auto* app = static_cast<DamathZero*>(appstate);
  delete app;
}
