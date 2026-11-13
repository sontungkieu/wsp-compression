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

The runtime configuration supplies `source`, `analysis_source`, two `lanes`
containing `gpu` and `uuid`, local `models`, expected `model_sha256` mappings,
`input_files` and `base_spec`. Consult `make_spec` in `campaign.py` and the
specification reads in `recovery.py` for all experiment fields.

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
