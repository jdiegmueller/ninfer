#pragma once

// Per-request wire statistics block shared by the protocol layers. The `timings`
// object mirrors the field names llama-server emits next to its OpenAI-compatible
// responses, so frontends that already parse that shape (e.g. llama-swap) ingest
// NInfer rates, cache reuse, and speculative counters without backend-specific
// handling. It carries no generated content and is emitted only for completed
// generations.

#include "serve/generation_service.h"

#include <nlohmann/json.hpp>

namespace ninfer::serve {

// Compose the per-request `timings` object from a finished generation:
//   prompt_ms / prompt_n / prompt_per_second
//       prefill-phase wall time (ms) and the rate of computed (non-cached)
//       input tokens; cached prefix tokens are excluded from the rate;
//   predicted_ms / predicted_n / predicted_per_second
//       decode-phase wall time (ms) and the rate of committed completion tokens;
//   cache_n
//       prompt prefix tokens reused from the resident cache;
//   draft_n / draft_n_accepted
//       speculative tokens drafted / accepted; present only when a speculative
//       backend ran for the request.
nlohmann::json make_wire_timings(const GenerationOutcome& outcome);

} // namespace ninfer::serve