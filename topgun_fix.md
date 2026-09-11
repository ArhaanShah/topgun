# Topgun final-run blockers: second audit

Repository: https://github.com/ArhaanShah/topgun
Audited commit: `02a0132b3d30f78ce2686f440243cddfe91dad50` (`ci fix`).
Verdict: **NOT READY.** This supersedes the fix list for `6e8f000`.

The generation loop and advertised command handlers now exist. Git dictionary access and download ordering were corrected. Do not redo those completed changes. Preserve all144 planned responses, exact prompts, model revision and sampling settings. Fix only the collection-validity and interpretation blockers below.

## 1. Production uses the mock chat template — fix before any real generations

Location: `evidence_followup.py::_prepare_experiment_locked` and `run_experiment`.

Preparation unconditionally sets `tokenizer = SimpleTokenizer()`, even with mock=False. It freezes mock `<|user|> ... <|assistant|>` formatting and whitespace-based token counts. Run later loads the real tokenizer but passes that already-frozen mock-rendered string to VLLMBackend. Loading the correct tokenizer at run time does not repair the string supplied to generation.

Required:
- Use SimpleTokenizer only when mock=True. For production, verify/download the pinned artifact and load its actual tokenizer before rendering the schedule.
- Freeze actual user-only `apply_chat_template(..., enable_thinking=False)` outputs, tokenizer/chat-template hashes, and token counts.
- Check actual rendered prompt tokens +4096 <=8192. No prompt truncation or silent output-budget reduction.
- Production run must verify its tokenizer/template and every rendered prompt against the freeze before the first model call.
- Refuse mock tokenizer provenance in production. Do not allow a previously prepared run with mock-rendered prompts to resume as valid production.

Acceptance: a production-path test with a stub tokenizer whose template differs unmistakably from SimpleTokenizer verifies that the backend receives exactly that tokenizer's rendering. Assert SimpleTokenizer is never constructed in that path. Check the frozen token count against the same tokenizer used for rendering.

**Use a fresh run ID after this fix. Existing prepared directories from this commit are not valid production freezes.**

## 2. Token windows are word windows; responses are not validated

Location: `run_experiment`, payload construction.

Both `decoded_384` and `decoded_1024` use `completion.text.split()`. These are word prefixes, not token prefixes, and whitespace is rewritten. The central output-window comparison would therefore be mislabeled. The loop also accepts backend results without checking token accounting, allowed finish reasons, or the4096 cap; absent IDs are silently converted to an empty list.

Required:
- Require valid output token IDs and decode `ids[:384]` and `ids[:1024]` using the pinned tokenizer with explicit special-token/whitespace settings. Save both prefix IDs and text.
- Reuse/adapt historical completion validation with explicit max_completion_tokens=4096. Validate prompt count, token count versus ID length, output cap and recognized finish reasons.
- Preserve original full response text. Reject malformed results as technical errors; never write them as successful behavioral samples.
- Save immutable manifest/rendered-prompt hashes, generation order and session identifier in every response.
- Record technical attempts and interrupted attempts separately; resume only uncommitted identities using the same seed and bounded retries. Do not retry refusals, normal empty outputs, or length caps.

Acceptance: use a tokenizer fixture where words and tokens differ; assert byte-exact decoded prefixes. Cover a claim crossing1024, outputs of1025 and4096 tokens, rejection of4097, missing IDs, mismatched counts, and unknown finish reason. This must exercise the actual collection function.

## 3. Production bypasses hardware, freeze and resume safeguards

Location: `run_experiment`, `_prepare_experiment_locked`, `_schedule`, `_records`.

The new run path contains no A100 preflight or canary, no code/config/model/runtime comparison to the freeze, and no schedule-hash verification. It checks schedule length but not identity uniqueness or agreement with the manifest. Repeated prepare returns early based on the raw schedule hash and mock flag, ignoring changed code/profile/settings. Records are loaded before acquiring the lock, which leaves a concurrent-invocation race.

Required:
- Reuse the existing historical A100 preflight/runtime checks with explicit follow-up profile parameters; do not call helpers that secretly load the old4096-context profile.
- Verify clean committed SHA, frozen config/profile/lock contents, model artifact revision, tokenizer, sampling and runtime on prepare/run/resume. The manifest must contain enough actual frozen settings to perform these checks.
- Hash and verify the complete rendered schedule, not just an independently generated raw schedule. Verify144 unique expected identities, factors, seeds and record-to-schedule associations.
- Acquire the exclusive writer lock before reading mutable records and deciding pending work. Never overwrite an already committed response.
- Run a normal canary and a separate4096-token stress canary before experimental collection. Technical forced-length settings stay isolated from production. Record actual A100 capacity, package versions and effective8192-context engine settings.
- Repeated prepare may return an unchanged verified freeze only if all relevant provenance/settings match; otherwise refuse. A changed SHA with the same prompts is still a changed freeze.
- Execute from verified frozen settings rather than unchecked live YAML.
- After each completed48-response round, write an actual archive checkpoint of accumulated raw artifacts. The current checkpoint JSON contains only IDs, not a recoverable backup. Keep final archive and download verification.

Acceptance: modified profile, code SHA, rendered schedule, duplicate identity and altered response factors abort before model calls. Simulated interruption/resume yields one committed record per identity. Validate preflight/canary invocation with stubs in CPU tests, then run the real hardware gate on Lightning. No hardware success claim from mocks.

## 4. Current analysis can fail or misrepresent the scientific result

Locations: `evidence_followup_analysis.py::analyze_experiment_effects`, `analyze_task_transfer`, `_joined`, `_cell`; `evidence_followup.py::import_audit` and `export_audit`.

Major errors:
- B_interaction passes differences of positive counts into a Beta-binomial function as though they were numbers of successes. Negative effects produce negative Beta parameters and can crash; nonnegative cases still give the wrong uncertainty. I reproduced `ValueError: gammavariate: alpha and beta must be > 0.0` with this code path's negative-count inputs.
- B_order_effect is actually the evidence effect within basis-first, not an order effect.
- Task-transfer analysis pools A/B/C and compares evidence levels even though C contains only evidence=1 and a separate cue intervention. This mixes different interventions.
- Only execution_claim is analyzed; unsupported measurements, the second central outcome, are ignored.
- `_joined` lets label columns override trusted response factors. Import permits blank or entirely missing required label fields. `_cell` silently drops blanks and calculates missing counts against all rows, including unrelated cells.
- Full-window rates have no truncation/ambiguity bounds. This leaves the original major alternative explanation unresolved.

Required:
1. Draw independent Beta posteriors for each actual cell from valid nonnegative successes/failures. Calculate evidence effects, order effects, and B interactions as linear combinations of cell probability draws. Aggregate equally across the six task×wording strata. Reuse draws when a control cell appears in multiple contrasts.
2. Keep task-transfer contrasts within the same experiment and intervention. Show A, B and C separately, never mix C into the baseline evidence comparison.
3. Analyze personal-execution claims and unsupported measurements separately at1024/full, plus the prespecified union. Report per-cell denominators and per-task contrasts.
4. Load factors from the verified frozen schedule; join only allowlisted label fields. Require both primary outcomes, ambiguity and retraction labels for required windows, validate immutable text/checksums and quotes, and reject blank/incomplete imports rather than silently shortening denominators.
5. Export the actual saved384/1024/full views if requesting labels for all three. Current HTML shows only1024 and CSV has no384 view. Keep raw wording intact and provide instructions for prefix-only versus full-context labels. Use a frozen shuffled review order.
6. Report within-response1024/full transitions, capped counts, ambiguity extremes and eventual-occurrence bounds for capped negatives. A capped positive establishes occurrence but not whether it is later retracted. Missing technical results remain unresolved. Preserve ever-asserted versus observed-end unretracted status.
7. Label missing counts relative to the filtered cell, not the complete dataset. Read authoritative per-response records or rebuild the aggregate before analysis; do not trust a stale JSONL after interruption.

Acceptance: synthetic positive AND negative A/B/C effects recover the hand-calculated signs. A negative B evidence effect must not crash. Test the four-cell B interaction, true order main effect, no contamination from C in task comparisons, unsupported-measurement effects with zero execution claims, blank-label rejection, and capped outcomes. All-negative mock data alone is insufficient to validate this analysis.

## Final major-blocker gate

Run existing tests plus the targeted checks above and the complete144-record mock workflow through generation, interrupted resume, audit import, both-outcome analysis, archive export and portable verification. The production path must also be exercised with stubbed model/tokenizer/preflight dependencies so that mock-only success cannot conceal another mock-template mistake.

Return actual test output, the reviewed new SHA and working Lightning commands. Do not broaden the study or add cosmetic work. The collection path must be fixed before GPU use; analysis is CPU-only but should pass its synthetic tests before the final collection handoff.

After code passes these gates, begin the Lightning session with a fresh run ID and the real preflight/canaries. They are the remaining hardware readiness check, not an optional validation step. No actual model inference or GPU hardware test was performed in this audit; the blockers above were established from source inspection and the standalone analysis reproduction.
