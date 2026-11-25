# Fixed-mask model correction experiments

This repository runs paired experiments on structured Qwen2.5 MLP pruning.
It compares mean-output compensation, ridge reconstruction, affine ridge and
candidate corrections proposed by Codex or a seeded random search.

The eight-hour campaign assigns one B200 to each search procedure. Each
procedure receives twenty candidate evaluations on the same validation data,
then freezes a candidate before reserved test scoring. The scripts persist
per-example scores, per-layer reconstruction diagnostics, parameter counts,
mask hashes, input hashes, runtime details and process receipts.

Read [EIGHT_HOUR_PLAN.md](EIGHT_HOUR_PLAN.md) for the study design, candidate
family, selection rule, reserved data regions and resource limits.

## Files

- `recovery.py` fits corrections and evaluates one explicit JSON specification.
- `campaign.py` launches two detached GPU workers with a shared eight-hour deadline.
- `analyze.py` computes paired document-bootstrap intervals for reserved test shards.
- `inspect_campaign.py` reads compact progress and verifies result hashes.
- `stage_data.py` downloads pinned LAMBADA and ARC-Easy files and verifies hashes.
- `runtime_config.example.json` shows all runtime configuration fields with portable example paths.
- `agent_round1.json` contains twelve initial Codex proposals and their rationales.
- `agent_round2_fallback.json` contains eight fallback proposals. A live feedback-based
  batch can be supplied as `agent_round2.json` in the campaign output directory.

## Runtime

The scripts use an existing Python 3.12 environment with PyTorch, Transformers,
NumPy and locally staged safetensors checkpoints. The exercised runtime uses
Transformers 5.3.0 and a CUDA-enabled PyTorch 2.10 development build. Model and
dataset downloads are staged separately. Execution uses offline Hugging Face
mode. Machine paths and GPU UUIDs are supplied in a private runtime configuration.

Run the analytical solver check with

```sh
python3 -B recovery.py --self-test
```

Run one experiment with an assigned GPU matching the JSON spec's `gpu` field

```sh
CUDA_VISIBLE_DEVICES=0 python3 -B recovery.py --spec spec.json --output result.json
```

Start a campaign after preparing `runtime_config.json` and placing both proposal
files beside it

```sh
python3 -B campaign.py --config runtime_config.json --root /absolute/new/campaign --mode start
python3 -B inspect_campaign.py /absolute/new/campaign
```

If a lane failed because its GPU acquired another compute process before any
holdout job started, the frozen selection can be continued in a fresh output
root. The worker waits for the original assigned GPU to be free and keeps the
original deadline. It does not repeat the search or interrupt other processes.

```bash
python3 -B campaign.py --mode resume --lane 0 --resume-from /absolute/original/campaign/lane0 --root /absolute/new/continuation
```

The runtime configuration supplies `source`, `analysis_source`, two `lanes`
containing `gpu` and `uuid`, local `models`, expected `model_sha256` mappings,
`input_files` and `base_spec`. Consult `make_spec` in `campaign.py` and the
specification reads in `recovery.py` for all experiment fields.

The separate `six_gpu_campaign.py` executes a seven-hour robustness grid on
GPU indices 2–7. It reads the earlier plan and frozen selection, verifies the
original source, dataset and model weight hashes, maps the documented storage
move, and records the resulting job specs and device UUIDs. Each independent
worker checks GPU occupancy before each job and stops its lane on failure. Its
120 jobs vary mask samples, fitting offsets and sizes, model, pruning fraction,
and context. They reuse previously observed test regions and are diagnostics,
not fresh confirmatory tests. `canary_six_gpu.py` checks the current runtime
with one small matched experiment before the full campaign.

`supplement_campaign.py` uses two assigned GPUs per host for a factorial
ablation of mask ranking (activation energy or variance) and ridge
regularization. It records independent output roots, reuses the parent plan's
frozen selection and deadline, and verifies the local GPU UUIDs before launch.

`functional_campaign.py` launches the compression, repetition, or control
queue on explicitly assigned GPUs under a shared deadline. It pins the original
data and checkpoint hashes, and each job fits physical weight patches from
finite difference responses. `summarize_functional.py` audits the completed
job receipts and reports paired outcomes by condition.
Follow-up stages compare logit readouts, calibration coverage, subspace design,
and calibration from model trajectories. The launcher can wait for a preceding
queue on the same GPU UUIDs. `stage_math_assets.py` stages the pinned Qwen3.5
and GSM8K assets via the configured HTTPS proxy. `natural_loop.py` uses a
separate train shard to locate natural loops and scores held-out math examples
after applying sparse projection or fitted patches.

`loop_gradient.py` compares one-token and sampled trajectory gradients for
selecting columns to attenuate. It evaluates the original weights, both
gradient selections and a layer-matched random selection at a common edit
norm. Its paired held-out rollouts record recurrence, output length, clean
next-token KL and a prompt-based copying probe. The WikiText motif cases are
controlled synthetic continuations.

`gradient_campaign.py` runs those comparisons independently on each assigned
GPU. A lane can wait for an explicit predecessor status file. Optional
offload mode checks the assigned UUIDs twice before admission, uses a
30-minute polling cadence and yields if another process appears on that GPU.
The watcher and each lane record absolute deadlines and write status JSON to
a fresh run directory. Both scripts support `--help`.

### Fixed-budget Deletion Tuning

`loop_deletion.py` compares static binary mask learning, three random masks,
and an Astra-style MLP output projection on Qwen2.5-0.5B and 1.5B. The base
arm preserves the checkpoint as the reference. Calibration uses synthetic
WikiText motif prefixes. Test prefixes come from a separate file.

- Use the same number of changed BF16 scalars per layer for every edited arm.
  The resulting Astra-style projection sets that budget for each condition.
  Track the realized Frobenius norm for each arm.
- Train binary mask scores with a fixed top-k forward mask and a sigmoid
  gradient. The loss discourages up to four consecutive motif tokens, with
  a reference KL term on clean and explicit-copy prompts. Learned and random
  masks share the same uniformly sampled pool of nonzero weights.
- Start in `gate_proj`, following the layer family in [Deletion Tuning](https://arxiv.org/html/2611.10866#S4).
  Repeat in `down_proj` to compare deletion and projection in the same matrix
  family. The original paper trains smoothly annealed gates with task losses.
- Evaluate free greedy and sampled rollouts. Save recurrence, motif
  continuation, output length, stop rate, clean KL and exact copy. Report
  paired prompt-bootstrap intervals against the base, Astra-style projection,
  and mean of the random masks. Evidence from these probes is scoped to
  controlled repetition on the benchmark.

Run one condition with `CUDA_VISIBLE_DEVICES` set to its assigned GPU and
`python3 -B loop_deletion.py --model MODEL --train TRAIN --test TEST --output RESULT`.
For a deadline-bounded GPU allocation, use `gradient_campaign.py --experiment deletion`
with `--payload-sha` for `loop_deletion.py` and one `--support-sha FILE=SHA256`
argument each for `loop_gradient.py`, `six_gpu_campaign.py`, and `wsp_functional.py`.
The queue begins with a small canary, followed by gate and down-projection
conditions at two budgets and three motif lengths. Use the host, GPU UUID,
available time, and immutable parent plan from the active allocation.

`insight_campaign.py` defines a ten-hour, eight-GPU follow-up queue. Two
lanes sweep multi-token copy protection by KL or exact-token likelihood.
Two lanes test motif-period and prefix-repetition transfer. Four pairwise
matched lanes compare loop-only editing and a direction derived from the end
of training rationales, across GSM8K subsets and decoding conditions. Natural
results also track answers given after `</think>`. The synthetic lanes record
an additional random control matched by per-layer magnitude strata. The
launcher pins input and source hashes, records each deadline and maintains
`audit.json` from validated result receipts. Run `--help` for the required
host, GPU UUIDs, deadline, parent plan, and natural asset manifest.

The earlier one-GPU campaign completed all 82 planned conditions in about
3 hours 15 minutes with zero failed jobs. Its recovered metric summary is
available in `night6h_result_summary.json`.

### Repetition diagnosis on original weights

`repetition_baseline.py` evaluates unmodified local checkpoints on a pinned
GSM8K panel, closed-set enumeration and exact intended repetition. It records
raw completions, lexical recurrence, a verifier outcome and native stop/cap
information. Detector matches are exploratory until audited against human or
task-state labels. Run `python3 -B repetition_baseline.py --self-test` for the
offline checks and `--summarize PATH` for per-panel counts.

`repetition_baseline_queue.py` schedules four independent baseline lanes on the
GPU UUIDs and deadline of the explicitly named predecessor. It runs small
canaries before paired panels and keeps partial rollout logs. The optional
`--retire-pilot` flag stops only the verified predecessor pilot workers on
those lanes, after the recorded natural-runtime failure. The supervisor may
use system Python and pass the CUDA runtime through `--worker-python`. A
continuation can select lanes with `--gpu` and wait for a named baseline run
with `--after-baseline`. Both commands support `--help`. The queue measures checkpoint and decoder comparisons before any
causal interpretation or targeted weight edit.

The pinned Qwen3.5-0.8B checkpoint supports `--astra-math --mode think
--decoder astra` with Transformers 5.3.0 and the staged model and GSM8K
manifest. This uses the released boxed-answer prompt and sampling settings
in a Transformers text-only evaluation with exact numeric scoring. The
baseline saves nested cap outcomes and verifies the checkpoint hashes.

`repetition_replay.py` selects exact terminal cycles from completed Llama
GSM8K runs after validating their digests and the pinned panel. It keeps
the original token prefix, compares greedy and nucleus sampling, and
compares a verified intermediate calculation with a token-length-matched
placebo. `--inventory` is a CPU-only check for the case gate, and
`--summarize` reports prompt-weighted continuation metrics from the raw
JSONL. Use `--astra` to select the two pinned Qwen3.5 Astra sampling shards
and replay with the original presence-penalty history. Both scripts support
`--self-test`.

Use `--astra --ablate-note PRIOR_REPLAY_DIR` to compare whitespace-only input,
a prompt to continue solving, and a prompt to give the final answer. All three
conditions reuse the exact prefix and the parent replay's per-prompt insertion
length, sampling settings, and generation cap. `--draw-start` and `--draws`
split the original draws into disjoint shards. `--summarize-ablation FILE ...`
validates the paired matrix and reports prompt-level bootstrap contrasts.
Add `--ablation-design frame` to cross the presence of the context delimiters
with three single-token fillers (space, newline, period). Each condition
keeps the same total insertion length. The summary includes within-filler
frame effects and comparisons against the period control.
Use `--ablation-design components` to cross the tag words, bracket and slash
punctuation, and boundary newlines. This keeps the newline token as the
filler and adjusts only the length of the central filler to match the
insertion budget. Tag variants are tokenized separately. This design reports
factorial contrasts and one-component removals from the full tag pair.
Use `--ablation-design punctuation` to cross the opening `[]`, closing
brackets, and the slash at the closing site while retaining the boundary
newlines. At that site the four conditions are empty, `/`, `[]`, and `[/]`.
Use `--ablation-design closing-marks` to leave the opening site empty and
compare an empty closing site with `[`, `]`, `[]`, and `()`. All five use
the same boundary newlines and central newline-token filler at the original
insertion length. The summary compares the square pair with each control and
estimates the interaction of the two square-bracket characters.

`repetition_mediation.py` compares empty, opening-square, and round-pair
closing marks and exchanges the probability mass of `Wait` variants and
`</think>` between the empty and opening-square conditions. Each donor
and receiver pair follows the same generated history. The exchange preserves
the receiver's within-group relative probabilities and uses the original
top-k and top-p filters after mixing. It operates during reasoning, ending at
the first generated `</think>`. The calibration cohort retains its original
insertion budget, and the held-out cohort uses 41 inserted tokens in every
arm. The report includes prompt-paired intervals, exact-cycle recurrence,
EOS, correct answers, and strata for Wait loops and other loops.

`repetition_mediation_queue.py` launches detached, deadline-bounded
supervisors on the eight-GPU discovery and four-GPU calibration nodes.
It verifies baseline shards, freezes a 64-prompt holdout disjoint from the
discovery panel, and checks the calibration summary before distributing
held-out replay across both nodes. It runs the joint swap, separate
one-group swaps, and a longer-cap replay on a fixed 24-prompt subset.
Each node writes its own receipts under a fresh shared run directory.

`repetition_exit_queue.py` schedules a fixed, independently verified
loop and nonlooping cohort across the two allocated B200 workspaces.
It tunes a fixed `</think>` logit bias on the prior cohort, measures
causal last-position residual swaps on that cohort, and gates held-out
testing on both receipts. `repetition_exit.py` compares an unmodified
prefix, length-matched closing-mark controls, direct probability transfer,
constant bias, and equal-duration fixed and loop-triggered pulses.
It reports accuracy, EOS, and exact-copy preservation separately by
stratum. The queue ends with a larger-cap replay and tests the selected
prefill-layer intervention with a matched-norm random control. Phase
progress and digests are recorded separately by supervisor.

For the Astra prompt reproduction, `repetition_baseline.py` now
validates the rendered canonical math instruction before tokenization.
`--astra-legacy-audit` explicitly adds the historical backspace
variant to the canonical run, using the same problem IDs and sampling
seeds. Save both forms with the paired queue in
`repetition_canonical_queue.py`, which reports full-panel differences
and freezes the replay cohort from canonical traces only.
`repetition_canonical.py` measures exact format separately from
numeric correctness, and uses explicit boxed, labelled, or wholly
numeric answers. It tests the matched bracket and probability-transfer
controls, then runs the block-4 patch/forced-tag comparison when the
canonical replay supports it. Both forced conditions process the
generated close tag normally and receive the same post-tag budget.
`repetition_exit_audit.py` rescores the historical rollouts using
only the known backspace-marker repair and reports paired contrasts
without editing the original artifacts.

`repetition_state_data.py` freezes native Qwen3.5 prefix cohorts by
problem ID. It distinguishes verified loops, states before their onset,
and ongoing reasoning without a detected cycle at the cut.
`repetition_state_value.py` compares native continuation, closing on the
next token (E0), and allowing up to 32 reasoning tokens before closing
(E32). E0 and E32 share the post-close budget and per-problem/draw
sampling stream. The success endpoint requires a correct numeric
answer, EOS, and no terminal cycle. Exact boxed formatting and
cap exhaustion are counted separately. The copy benchmark uses the
Qwen chat turn terminator and scores the literal visible answer.

`repetition_rank_one.py` learns a block-4 residual direction from
matched dev prefixes and tests it on native prefixes. When that
direction improves development success, `repetition_rank_fit.py` fits
a no-intercept ridge gate to positive E0–E32 advantage with penalties
at other token positions. `repetition_rank_eval.py` compares the gate
with a bias calibrated on dev to a similar first-token close rate.
All-token hooks and transient BF16 edits are gated on held-out success
and checked against parity and a matched random edit. The supervisor
`repetition_state_queue.py` records worker receipts and enforces its
deadline. Its `--copy-only` mode runs copy calibration and held-out
copy generation under a fresh output root. To test sensitivity to the
timing cap, freeze an eval-only cohort with a `labels_cap` between 512
and 2048, run the timing smoke on that cohort, and use `--eval-and-copy`.

`repetition_sampler_queue.py` runs the native sampler reachability
diagnostic. Its cohort builder keeps the dev panel separate and selects
fresh canonical prefix IDs before falling back to previously reported
holdout IDs. Prefill compares native, blank, and bracket contexts and
last-token donor patches using the original presence penalty, temperature,
and Transformers filters. It records verified brackets for entry into
support and marker probability 0.5. The queue freezes up to two native
activation candidates on dev, then runs held-out prefill and paired replay.
Replay compares each activation with a first-step decoder control that
matches the marker probability and preserves native non-marker conditionals.
The native and patched policies share 512-token reasoning and post-close
budgets. All runs record per-prompt hashes and shard receipts.
The optional `--norm-only` run compares directions at equal per-state
L2 budgets set by the blank-to-bracket and native-to-bracket distances.
It uses a separate output root and performs prefill only.

`repetition_budget_cohort.py` freezes a balanced panel of native GSM8K
states and a question-only subset before replay. `repetition_budget_run.py`
generates native, E0, E32, and E128 trajectories and records their common
trunks and EOS token IDs. Its two-GPU supervisor admits additional complete
draws based on the measured wave duration and hard deadline.
`repetition_budget_score.py` cuts the same stored trajectories at equal
post-marker or equal total continuation budgets, counting the closing
marker and EOS within total windows. It reports prompt-paired intervals,
answer status and strict formatting, and can also rescore the archived
1,536-token timing or 512-token reachability trajectories.
An extension manifest prepared with `--extend-from` freezes eight IDs per
stratum. The supervisor uses that manifest for one paired E0/E128 draw at
3,072 post-marker tokens, and the scorer evaluates nested 2,048 and 3,072
cuts on those same trajectories.

`repetition_scratchpad_cohort.py` audits prior GSM8K IDs, allocates disjoint
canonical discovery panels when needed, and freezes exact source scratchpads.
`repetition_scratchpad_queue.py` runs discovery and four paired E0 draws on
the frozen panel. A 2×2 intervention keeps or drops the context suffix
independently of the sampler's initial presence set. In the dropped-context
arm the preserved set conditions only the sampler. All arms keep the same
presence penalty on newly generated tokens, the same post-marker seed, and
the same padded width per batch.
`repetition_scratchpad_score.py` verifies the per-arm hashes and reports
numeric correctness with EOS from nested 256- and 512-token windows, with
prompt-paired contrasts and the factorial interaction.
`repetition_scratchpad_cap.py` freezes a hash-ranked subset of the factorial
panel for 2,048-token cap sensitivity with reset presence history. It audits
cross-run 512-token trace parity separately from the nested cap comparisons and
can tabulate paired terminal outcomes and revision of explicit boxed answers.

`repetition_restart_cohort.py` freezes disjoint pilot and evaluation panels of
numeric-answer problems. `repetition_restart_queue.py` compares native thinking,
native no-thinking with the same sampler, a same-template immediate exit, and
fixed-token switches with retained or discarded reasoning. The switch arms
both reset presence history and preserve native outcomes when the model closes
before the switch. Discarded reasoning still counts against the total budget.
Pilot generation checks that a real restart matches a nested immediate-exit
trace at the remaining budget before evaluation can reuse the nested trace.
`repetition_restart_extension.py` measures nested 2,048- and 4,096-token
windows on a hash-ranked 32-problem subset, keeping the shared-sampler
comparison and a separate no-thinking control configured from the pinned
Qwen model card. `repetition_restart_visible.py` additionally scores the
explicit assistant turn boundary independently of the configured EOS.

`repetition_cache_swap_queue.py` tests the joint convolution and recurrent
state swap at the first forced closing token on Qwen3.5. The canary checks
that the attention cache bytes stay identical up to that forward call, that
donor states are copied, and that self-copy reproduces the native token trace.
The development score compares the swap with native retained context and the
question-only reference at the same post-marker budget.

`repetition_restart_transfer.py` repeats the question-level matrix on
Qwen3-1.7B at 4,096 tokens, with a nested 2,048-token cut on the same 32
questions as the Qwen3.5 budget extension. Its preflight verifies bytes
against the public checkpoint snapshot and derives closure, turn and EOS IDs
for that tokenizer. The canary checks explicit restart, the common trunk,
and the model-specific token ledger on a separate panel.

`repetition_cache_factorial.py` reuses each of two native 512-token traces
on the 32 reserved numeric MATH-500 questions. It tests convolution and
recurrent matrix state changes independently at the forced closing token,
verifies identical initial attention cache bytes across the four treatments,
and reports factorial effects and the question-only reference at the same
continuation budget.

`repetition_published_astra.py` verifies the original and edited BF16
tensor digests for the published sparse Astra patch before comparing native
generation and forced exits on matched reasoning prefixes. The worker applies
the patch to the loaded model in memory and pairs it with the original model
on math and a separate no-thinking copy set.

`repetition_exit_scaffold.py` varies newlines immediately before and after
the closing marker. It checks the rendered no-thinking bytes and couples the
presence set, post-marker seed, and answer budget across all four variants.
The canary compares a prefilled marker with a forced marker at an identical
token sequence and records logit drift.

`repetition_phase_trace.py` reports sustained exact cycles, recurrent
16-grams and within-window duplicate fractions before the reasoning close.
Its switch analysis uses only tokens available at the fixed switch point.
`repetition_native_horizon.py` chooses a long rollout cap for 12 fixed
questions after a separate timing pilot and scores nested windows from
2,048 tokens up to the selected 16,384- or 32,768-token horizon.
`repetition_late_replay.py` freezes two cut points per still-open trace
and compares native continuation, forced closure, convolution-only refresh,
joint recurrent-state refresh, and question-only restart. It charges the
marker and EOS to one continuation budget and audits the original motif at
the end of each continuation, along with the marker's sampling probability.

`repetition_cache_retention.py` changes one checkpoint value in a
token-length-matched synthetic scratchpad while holding the question fixed.
It measures whether each cache treatment tracks both values in exact copy
and arithmetic, and verifies that the question-only reference sees an
identical input and produces an identical trace for each value pair.
The copy report includes an EOS diagnostic for a correctly mentioned nonce.
`repetition_cache_content_score.py` contrasts correct EOS for the observed
value with correct EOS for the opposite value, so an identical lucky guess
credits neither cache treatment nor the question-only reference.

`repetition_astra_source_reader.py` crosses base and Astra scratchpads with
base and Astra readers under the same forced closing token and residual
budget. It freezes eligibility at each landmark, checks identical-source
response parity, and separates the scratchpad source effect from the reader
checkpoint effect.

`repetition_checkpoint_replay.py` tests cache interventions on intermediate
states written by the model. A simulator verifies the third step before the
replay landmark. The analysis charges the reasoning tokens to the total
budget and reports both eligible-state performance and full-panel policy
performance with native fallback.
The optional step-context format accepts a labeled state or a completed
equation inside step 3. It validates coverage on a fresh pilot before
evaluating the original held-out questions.

`repetition_model_sizes.py` checks native thinking and no-thinking on the
same MATH-500 questions across byte-verified Qwen checkpoints. It scores
2,048 and 4,096 tokens from each rollout and uses model-specific closing and
EOS tokens when measuring completion and repetition.

`repetition_late_budget.py` extends the observed late-replay panel to 4,096
continuation tokens and compares nested windows with a no-thinking run from
the question. A teacher-forced probe measures the probability of one complete
cycle under the original presence history.

`repetition_qwen35_horizon.py` compares longer thinking rollouts for
Qwen3.5-0.8B and Qwen3.5-2B on new numeric MATH-500 questions. It selects a
common native cap from a disjoint timing pilot and limits the separately
reported no-thinking baseline to 4,096 tokens.

`repetition_late_replication.py` freezes a held-out replay cohort only when
the new horizon yields at least four exact cycles. Its primary outcome is
the paired convolution refresh effect at 1,024 continuation tokens. A
secondary policy spends the unused budget on a direct answer when the
convolution replay reaches 1,024 tokens without EOS.

`repetition_late_confirmation.py` locks a fresh numeric MATH-500 panel,
checks the runtime against a disjoint timing pilot, and runs the original
horizon and replay workers. It scores continuation accuracy on trigger
cases and simulates the whole-panel intervention at a fixed total token
budget when every trigger case has verified replay results.

`repetition_external_panel.py` converts a checksum-pinned public numeric
dataset such as SVAMP or Numina AMC into the follow-up panel schema. It
preserves the dataset revision and split, excludes exact normalized
problem matches from supplied source files and archived cohort manifests,
and freezes disjoint development and evaluation IDs before generation.

`repetition_followup_queue.py` runs independent, checksum-pinned GPU and CPU
jobs from append-only manifests. The configuration pins the host, device
UUIDs, runtime and the access and compute deadlines. Each completed shard
has a receipt. The supervisor records GPU release and realized GPU hours.
For a subset allocation, record the complete ordered host roster in
`host_devices` and the assigned UUIDs in `devices`. Scheduling and release
receipts use the assigned UUIDs.

`repetition_campaign_resume.py` restores the interrupted Nemo campaign in a
fresh run directory. It checks the archived manifests and completed dependency
receipts before reusing them. `repetition_campaign_handoff.py` seals the resumed
queue, starts its reporter and stages a separate Huy successor that waits for
the predecessor's release receipt and an idle device roster. These two
scripts contain the campaign-specific host identity, archive pins and cutoff.

`repetition_followup_baseline.py` runs native thinking and direct decoding
on the same prompts. Its agreement policy gives two draws half the total
token budget each. `repetition_followup_timing.py` replays native prefixes
at locked positions or at the first confirmed exact cycle, with token
cost counted from the original question. It tests keeping the prefix,
refreshing convolutional cache state or answering directly from the question.
The optional zero-convolution replay measures whether a zero reset reproduces
the effect of restoring the question-only convolutional state.

`repetition_followup_information.py` crosses two hidden checkpoint values
with two operations and scores all four outputs against exact numeric
answers. Its optional zero-convolution arm clears only the convolutional
state at the marker and verifies that the attention KV remains unchanged.
`repetition_followup_evidence.py` reuses the original fresh-evidence
trajectories and pairs them with a direct instruction to use the same record.
`repetition_followup_astra.py` verifies exact published BF16 tensor
replacements and compares native and fixed-marker math with a separate
copy and formatting check. `repetition_followup_plan.py` admits the
external transfer matrix, and `repetition_followup_aggregate.py` combines
verified scores and reports paired effects and direct-answer risk and coverage.
`repetition_followup_audit.py` estimates the paired convolution-by-matrix
interaction and compares timing policies on prompts eligible at both anchors.

`repetition_information_depth.py` scores nested continuations on the original
hidden-checkpoint panel and reports numeric answers before EOS separately. It
compares the original evidence text with a direct instruction to use the same
record. `repetition_motif_plan.py` pins archived sources for one-step motif
interventions. `repetition_depth_control.py` schedules a successor only after the
original run releases every assigned device. `repetition_depth_aggregate.py`
combines the resulting score receipts, including the paired depth cuts.
`repetition_delayed_query.py` freezes paired synthetic records and reveals a
read or arithmetic query only after prefix prefill. It tests donor convolution
reset, intact cache and fresh-record controls at the same post-marker cap.
Its pilot summary gates the held-out run on signed record use with exact
question-only pair parity.
`freeze-match` retains the original cases and apply query and extends the
read query to the same token length. The matched evaluation compares keep
against reset-conv at the 256/512 cuts of one trace. `summarize` reports the
paired correct-plus-EOS interaction and optionally checks it against the
checksum-pinned original score shards.
`freeze-match --padding whitespace` inserts ordinary whitespace tokens
after the first read token while preserving the entire read clause. Its
separate pilot checks read accuracy before admitting the holdout.
`freeze-after` verifies an existing before-padding panel and moves its exact
padding block after the same read clause. The paired summary compares the
effect of reset-conv across the two padding positions.
`freeze-after --front-padding-tokens 6` keeps the first six padding tokens
before the clause and moves the last six after it. The length audit with
`--after-reference` compares 6+6 against both 12+0 and 0+12 using the same
checksum-verified, completed question shards at both token caps.
`--split-tail PATH` adds the final two four-question shards from a separate
released run, preserving the original score and release receipts.
`repetition_delayed_timing.py` uses the same 6+6 READ token stream and
replaces only the convolution cache before forced token 0, 7, or 17.
Each public donor has consumed the identical forced query prefix at the
capture position. The scorer retains the nested 256/512 correct-plus-EOS
endpoints and reports first-answer distribution changes at the marker.
`repetition_delayed_transport.py` restores KEEP cache components at offset
7 or 17 after an early convolution swap. Full restoration must reproduce
the KEEP logits exactly. Partial restorations report first-answer total
variation with paired confidence intervals.
`repetition_delayed_numeric.py` teacher-forces both candidate numerals after
the same newline and reports paired correct-versus-opposite log odds and
the total probability mass of the two numeral continuations.
`repetition_delayed_value_cross.py` swaps the batch rows of convolution,
recurrent matrix and attention caches independently on value-only record
pairs. All three swaps together must reconstruct the opposite row's
conditional numeral scores.
`repetition_delayed_replication.py` freezes new record values and IDs while
preserving the archived 6+6 token layout and family-by-slot balance.
`repetition_delayed_restore_rollout.py` measures correct-plus-EOS after
selective restoration at a fixed offset on a held-out READ panel. The value
cross can also compare a byte-pinned hybrid checkpoint on the same examples.
For the same checkpoint, `--offset 17` crosses values after the READ clause
and can compare its component effects with checksum-pinned offset-0 shards.
The numeric parser stops at the assistant turn marker, while success still
requires the configured EOS inside the token budget. `rescore` uses the
original checksum-pinned rollout worker to validate archived raw traces.
`repetition_delayed_campaign.py` pins the cached artifacts and launches the
two canaries, the 16-ID pilot, and the gated 64-ID holdout on the assigned
General Embedding node. Its snapshot includes live GPU ownership and release
evidence. `repetition_delayed_results.py` checks completed shard receipts and
reports cluster-paired contrasts by query and record template.
`repetition_delayed_length_campaign.py` pins the original score receipts and
limits the length-matched follow-up to GPU indexes 1 and 3. The holdout starts
only after both canaries recover a recorded value with the extended READ query.
With `--padding whitespace`, the queue also requires the frozen 16-case pilot
to recover at least 24 of 32 READ values with EOS. Its final comparison uses
new CPU scores derived from the original frozen raw trajectories.
With `--padding after --before-panel PATH`, the queue moves the exact frozen
pad after READ, uses technical canaries without an accuracy threshold, and
compares the same holdout against the before-padding score receipts.
`--padding split` keeps six frozen tokens on each side of READ, uses the
first two holdout shards as the technical smoke, and admits the remaining
shards with the shorter measured reserve under the same hard cutoff.
`repetition_delayed_split_tail.py` verifies the seven archived split shards
and starts only cases 56–63 in a fresh run on a newly allotted window.
`repetition_delayed_timing_campaign.py` holds the 6+6 READ prefix fixed and
admits the reset-offset holdout after two technical checks with no accuracy
threshold. `repetition_delayed_group_audit.py` describes archived READ
completion by record family and queried slot.
`repetition_delayed_timing_audit.py` checks READ token traces against the
archived KEEP and reset-at-zero controls. The transport campaign admits its
restoration checks after the timing campaign's verified GPU release.
`repetition_delayed_numeric_campaign.py` queues the bare-numeral scoring probe
on all three archived query placements after the restoration release.
`repetition_delayed_value_cross_campaign.py` queues the paired cache crossing
after the numeral probe. `repetition_delayed_probe_chain.py` checks every
launcher's source bytes and waits for predecessor release and GPU idleness
before admitting the next probe within the allocation.
`repetition_delayed_replication_campaign.py` admits an independent timing or
restoration holdout on disjoint balanced record panels after their verified
predecessor releases. `repetition_delayed_transfer_campaign.py` verifies the
cached 2B checkpoint and compares crossed cache values on matched 0.8B cases.
Its `--offset 17` mode keeps the 0.8B checkpoint and compares each crossed-cache
effect before and after the fixed READ clause.
`repetition_delayed_final_chain.py` queues these two comparisons after the
restoration holdout, with cohort size determined by time remaining.
`repetition_delayed_value_audit.py` joins verified 0.8B readout and crossing
shards and counts paired numeral preference changes by record template and slot.
`repetition_delayed_length_audit.py` verifies unchanged APPLY prefixes, seeds
and exact continuation tokens against checksummed shards. Its
`--position-effects` option reports paired READ differences by cache arm.
`repetition_delayed_length_rescore.py` rechecks both archived READ/APPLY
matrices using the first assistant turn delimiter as the answer boundary
while preserving the configured EOS requirement and the original token cuts.
`repetition_motif_path_audit.py` checks the first confirmed return to the
original motif and natural phase closure at period-scaled token budgets.
For input produced by the extended baseline worker, the motif replay
requires `--source-worker baseline_extended` and verifies that worker's
checksum before applying its existing raw receipt checks.
`repetition_motif_intervention.py` compares native continuation, one
off-cycle token inside thinking, forced closure and direct decoding with
the same continuation budget. Its `run` action verifies the archived
baseline and writes raw trajectories. `score` checks their provenance and
reports task outcomes, motif relapse and token cost.

`repetition_motif_guard.py` replays the same one-token intervention with
and without blocking completions of any rotation of the original motif for
two periods. The guard redistributes ordinary token mass and preserves the
local probabilities of phase closure and stopping. Its `score` action
checks branch parity and records both motif return and downstream outcomes.

## Interpretation

ARC-Easy uses an explicit zero-shot answer-continuation likelihood protocol.
LAMBADA uses left-truncated context and exact greedy target-token matching.
Scores are reported with their custom protocols. The public study currently
has one Codex trajectory and one random-search seed. The later transfer grid
reuses heldout regions after selection and is interpreted as diagnostic.

Mean compensation and activation-based pruning have established prior art,
including [FLAP](https://arxiv.org/html/2312.11983). The weight-space-programming
motivation comes from [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/).
The experiment measures quality recovery and the performance of two bounded
search procedures within a specified correction family.
