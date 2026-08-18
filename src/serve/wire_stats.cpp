// Per-request `timings` wire block. See wire_stats.h for the field contract.

#include "serve/wire_stats.h"

#include <algorithm>

namespace ninfer::serve {

nlohmann::json make_wire_timings(const GenerationOutcome& outcome) {
    const int prompt_tokens     = outcome.prompt_tokens;
    const int completion_tokens = outcome.completion_tokens;
    const int cached_tokens =
        std::clamp(static_cast<int>(outcome.metrics.prefix_cache_hit_tokens), 0, prompt_tokens);
    const int computed = std::max(0, prompt_tokens - cached_tokens);
    const double prefill_s = outcome.metrics.prefill_seconds;
    const double decode_s  = outcome.metrics.decode_seconds;
    const double prompt_rate    =
        (computed > 0 && prefill_s > 0.0) ? static_cast<double>(computed) / prefill_s : 0.0;
    const double predicted_rate =
        (completion_tokens > 0 && decode_s > 0.0)
            ? static_cast<double>(completion_tokens) / decode_s
            : 0.0;
    nlohmann::json timings = {
        {"prompt_ms", prefill_s * 1000.0},
        {"prompt_n", prompt_tokens},
        {"prompt_per_second", prompt_rate},
        {"predicted_ms", decode_s * 1000.0},
        {"predicted_n", completion_tokens},
        {"predicted_per_second", predicted_rate},
        {"cache_n", cached_tokens}};
    if (outcome.metrics.speculative_backend != SpeculativeBackend::None) {
        timings["draft_n"]          = outcome.metrics.speculative_draft_tokens;
        timings["draft_n_accepted"] = outcome.metrics.speculative_accepted_tokens;
    }
    return timings;
}

} // namespace ninfer::serve