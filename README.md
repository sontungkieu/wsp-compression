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

The earlier one-GPU campaign completed all 82 planned conditions in about
3 hours 15 minutes with zero failed jobs. Its recovered metric summary is
available in `night6h_result_summary.json`.

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
