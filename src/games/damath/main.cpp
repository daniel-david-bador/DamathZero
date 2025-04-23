#define SDL_MAIN_USE_CALLBACKS
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <torch/torch.h>

import std;
import alphazero;
import damathzero;

static SDL_Window* window = NULL;
static SDL_Renderer* renderer = NULL;

auto SDL_AppInit(void** appstate, int, char*[]) -> SDL_AppResult {
  SDL_SetAppMetadata("DamathZero", "1.0", "com.bads.damathzero");

  if (!SDL_Init(SDL_INIT_VIDEO)) {
    SDL_Log("Couldn't initialize SDL: %s", SDL_GetError());
    return SDL_APP_FAILURE;
  }

  if (!SDL_CreateWindowAndRenderer("DamathZero", 640, 480, 0, &window,
                                   &renderer)) {
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

  static auto model = alpha_zero.learn({
      .action_size = Damath::ActionSize,
      .num_blocks = 2,
      .num_attention_head = 4,
      .embedding_dim = 64,
      .mlp_hidden_size = 128,
      .mlp_dropout_prob = 0.1,
  });

  *appstate = model.get();

  return SDL_APP_CONTINUE;
}

auto SDL_AppEvent(void*, SDL_Event* event) -> SDL_AppResult {
  if (event->type == SDL_EVENT_QUIT)
    return SDL_APP_SUCCESS;

  return SDL_APP_CONTINUE;
}

auto SDL_AppIterate(void*) -> SDL_AppResult {
  const int charsize = SDL_DEBUG_TEXT_FONT_CHARACTER_SIZE;

  /* as you can see from this, rendering draws over whatever was drawn before
   * it. */
  SDL_SetRenderDrawColor(renderer, 0, 0, 0,
                         SDL_ALPHA_OPAQUE); /* black, full alpha */
  SDL_RenderClear(renderer);                /* start with a blank canvas. */

  SDL_SetRenderDrawColor(renderer, 255, 255, 255,
                         SDL_ALPHA_OPAQUE); /* white, full alpha */
  SDL_RenderDebugText(renderer, 272, 100, "Hello world!");
  SDL_RenderDebugText(renderer, 224, 150, "This is some debug text.");

  SDL_SetRenderDrawColor(renderer, 51, 102, 255,
                         SDL_ALPHA_OPAQUE); /* light blue, full alpha */
  SDL_RenderDebugText(renderer, 184, 200, "You can do it in different colors.");
  SDL_SetRenderDrawColor(renderer, 255, 255, 255,
                         SDL_ALPHA_OPAQUE); /* white, full alpha */

  SDL_SetRenderScale(renderer, 4.0f, 4.0f);
  SDL_RenderDebugText(renderer, 14, 65, "It can be scaled.");
  SDL_SetRenderScale(renderer, 1.0f, 1.0f);
  SDL_RenderDebugText(
      renderer, 64, 350,
      "This only does ASCII chars. So this laughing emoji won't draw: 🤣");

  SDL_RenderDebugTextFormat(renderer, (float)((680 - (charsize * 46)) / 2), 400,
                            "(This program has been running for %" SDL_PRIu64
                            " seconds.)",
                            SDL_GetTicks() / 1000);

  SDL_RenderPresent(renderer);

  return SDL_APP_CONTINUE;
}

auto SDL_AppQuit(void*, SDL_AppResult) -> void {}
