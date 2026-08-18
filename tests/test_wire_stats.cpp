// Contract test for the per-request `timings` wire block (make_wire_timings):
// rate arithmetic (computed tokens / prefill phase, committed tokens / decode
// phase), cache clamping, zero-phase guards, and the speculative-fields
// presence rule. The block mirrors the llama-server field names so frontends
// such as llama-swap can ingest the statistics from the response body.

#include "serve/wire_stats.h"

#include <nlohmann/json.hpp>

#include <iostream>

namespace {

using Json = nlohmann::json;
using namespace ninfer::serve;
using ninfer::SpeculativeBackend;

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int check(bool condition, const std::string& message) { return condition ? 0 : fail(message); }

GenerationOutcome base_outcome() {
    GenerationOutcome outcome;
    outcome.prompt_tokens   = 1000;
    outcome.completion_tokens = 128;
    outcome.metrics.prefill_seconds = 0.05;
    outcome.metrics.decode_seconds  = 2.0;
    outcome.metrics.prefix_cache_hit_tokens = 900;
    return outcome;
}

int test_rates_and_counts() {
    int failures = 0;
    GenerationOutcome outcome = base_outcome();
    const Json timings = make_wire_timings(outcome);
    failures += check(timings.at("prompt_n") == 1000, "prompt_n is full input length");
    failures += check(timings.at("predicted_n") == 128, "predicted_n is completion length");
    failures += check(timings.at("cache_n") == 900, "cache_n is the reused prefix");
    failures += check(timings.at("prompt_ms") == 50.0, "prompt_ms is prefill phase ms");
    failures += check(timings.at("predicted_ms") == 2000.0, "predicted_ms is decode phase ms");
    // Rate covers only the computed suffix: (1000 - 900) / 0.05 s.
    failures += check(timings.at("prompt_per_second") == 2000.0,
                      "prompt rate excludes cached prefix");
    // Committed completion tokens over the decode phase: 128 / 2.0 s.
    failures += check(timings.at("predicted_per_second") == 64.0, "decode rate from phase wall");
    // No speculative backend: the draft counters are absent, not zero.
    failures += check(!timings.contains("draft_n"), "draft_n absent without speculative backend");
    failures +=
        check(!timings.contains("draft_n_accepted"), "draft_n_accepted absent without speculative");
    return failures;
}

int test_speculative_counters() {
    int failures = 0;
    GenerationOutcome outcome = base_outcome();
    outcome.metrics.speculative_backend         = SpeculativeBackend::Mtp;
    outcome.metrics.speculative_draft_tokens    = 384;
    outcome.metrics.speculative_accepted_tokens = 320;
    const Json timings = make_wire_timings(outcome);
    failures += check(timings.at("draft_n") == 384, "draft_n is drafted token count");
    failures += check(timings.at("draft_n_accepted") == 320,
                      "draft_n_accepted is accepted token count");

    // A speculative run with zero acceptance still reports both counters.
    outcome.metrics.speculative_accepted_tokens = 0;
    const Json zero_accept = make_wire_timings(outcome);
    failures += check(zero_accept.at("draft_n") == 384 && zero_accept.at("draft_n_accepted") == 0,
                      "zero acceptance reports 0 accepted, not absent fields");
    return failures;
}

int test_zero_phase_guards() {
    int failures = 0;
    // Fully cached prompt: no prefill wall, no computed suffix.
    GenerationOutcome fully_cached = base_outcome();
    fully_cached.metrics.prefill_seconds = 0.0;
    const Json cached_timings = make_wire_timings(fully_cached);
    failures += check(cached_timings.at("prompt_per_second") == 0.0,
                      "no division by zero on empty prefill phase");
    failures += check(cached_timings.at("prompt_ms") == 0.0, "zero prefill phase ms");

    // Single-token completion: no decode phase wall.
    GenerationOutcome single_token = base_outcome();
    single_token.completion_tokens         = 1;
    single_token.metrics.decode_seconds    = 0.0;
    const Json single_timings = make_wire_timings(single_token);
    failures += check(single_timings.at("predicted_per_second") == 0.0,
                      "no division by zero on empty decode phase");

    // A hit count above the input length clamps instead of going negative.
    GenerationOutcome over_count = base_outcome();
    over_count.metrics.prefix_cache_hit_tokens = 2000;
    const Json clamped = make_wire_timings(over_count);
    failures += check(clamped.at("cache_n") == 1000, "cache_n clamps to prompt length");
    failures += check(clamped.at("prompt_per_second") == 0.0, "no computed suffix after clamp");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_rates_and_counts();
    failures += test_speculative_counters();
    failures += test_zero_phase_guards();
    if (failures != 0) { return 1; }
    std::cout << "test_wire_stats: OK\n";
    return 0;
}