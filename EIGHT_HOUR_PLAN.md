# Eight-hour fixed-mask correction study

## Question and resource budget

Compare a Codex-guided correction search against a seeded random search under
the same 20-candidate validation budget. Each lane has one B200 for at most
eight wall-clock hours. The total upper bound is sixteen GPU-hours. Each
candidate and each evaluation shard runs as a separate bounded job.

The initial model is Qwen2.5-1.5B base with 50% of each MLP's intermediate
channels physically removed. The mask is frozen from 4,096 training tokens.
This is approximately 37.4% of the original whole-model parameters. All
candidate arms retain identical shapes and bias slots. Correction fitting
uses 4,096 training tokens and ridge relative regularization 1.0, selected in
the preceding campaign.

## Correction family

Both search procedures can choose ridge or affine-ridge coefficients,
correction gain from {0.5, 0.75, 1, 1.25, 1.5}, a mean-bias mixing fraction
from {0, 0.25, 0.5, 0.75, 1}, one of six predefined layer profiles, and an
optional per-output-row update norm cap from {0.05, 0.1, 0.2, 0.4} times the
surviving weight-row norm. The six profiles are flat, front, back, middle,
protect-late and protect-early. They alter only the correction coefficients.

Codex proposes twelve initial candidates with recorded rationales, inspects
their validation feedback and supplies eight follow-up candidates. A frozen
eight-candidate fallback batch lets the queue continue if the browser session
ends before follow-up proposals arrive. The queue records the actual consumed
batch and its SHA-256. The random lane draws twenty distinct candidates using
seed 20261114 from exactly the same parameter choices.

Every lane also evaluates the identical ridge reference. Its cost is recorded
separately from the twenty search candidates. Every candidate receives the
same validation examples, model-forward inputs and fitting data. Both lanes
compute the same underlying ridge and affine systems before applying their
chosen correction.

## Validation and selection

Validation uses 64 WikiText windows at context length 512 and 128 ARC-Easy
validation questions. Each lane selects the candidate with maximum
length-normalized ARC accuracy among those whose WikiText NLL is at most 0.02
above the fixed ridge reference. NLL breaks accuracy ties. The identical
reference is retained when no candidate qualifies. The selected candidate and
rule are written to an immutable selection receipt before test scoring.

There is one Codex trajectory and one random-search seed in this run. The
result estimates the performance of these particular searches. Additional
trajectories are needed to estimate a general advantage of an editor model.

## Test and diagnostic sets

ARC-Easy test indices 1664 through 2375 and LAMBADA indices 512 through 1023
and 4352 through 5119 were reserved from the previous broad sweep. They are
scored only after the lane's candidate has been selected. Per-example records
support paired confidence intervals. Later transfer and robustness jobs reuse
these selected test regions and are reported as diagnostics.

WikiText test at offset 140000 has been observed previously and remains a
diagnostic measure. Existing custom answer-likelihood and last-word protocols
are retained for continuity. The machine has no installed lm-eval-harness, so
these scores are explicitly identified by their recorded custom protocols.

## Queue order

1. Run the shared validation reference and twenty search candidates per lane.
2. Freeze each selected candidate and score the reserved test sets in shards.
3. Check transfer to Qwen2.5-0.5B base, Qwen2.5-1.5B base and
   Qwen2.5-7B-Instruct at 25% and 50% MLP pruning.
4. Measure eager-forward prefill latency for each paired arm after warm-up,
   at batch size one and token lengths 256, 512 and 1024. The receipt identifies
   resident reference modules, measurement repetitions and CUDA-event timing.
5. Vary fitting samples across offsets 0, 8192 and 16384, using 256, 1024,
   4096 and 8192 tokens while keeping the mask fixed.
6. Check context lengths 256, 1024 and 2048 across 10%, 25%, 50% and 75%
   MLP pruning and the three checkpoints.
7. Spend remaining budget on larger paired test evaluations across calibration
   offsets 0, 4096 and 8192. Each job has a bounded timeout and retains completed
   arms when a later arm reaches the deadline.
8. If budget remains, measure the frozen candidate's sensitivity to ridge
   regularizers 0.01, 0.1 and 10 on the three checkpoints. These diagnostic
   comparisons retain the previously selected candidate structure.

Comparisons always include the dense model, pruning alone, mean compensation,
ridge, affine ridge and the selected candidate. Qwen2.5-7B-Instruct differs in
instruction tuning as well as size, so checkpoint transfer is reported directly.

## Execution and evidence

The supervisor starts two detached CPU workers, one per assigned GPU. Each
worker launches at most one GPU child at a time. Before every job it verifies
the assigned GPU UUID, free VRAM and absence of another compute process. It
verifies source and input hashes and official checkpoint file hashes. It
records PID, source/spec hashes, mask digest, wall time, peak allocated memory,
per-example scores and per-layer reconstruction diagnostics.

The eight-hour deadline is shared by both workers. The supervisor stops its
own child process group at a job timeout or deadline, preserves partial-arm
results and writes the final lane status. Encountering another GPU process
stops admission of further jobs on that lane. Completed work remains available
when the browser connection closes. Three failed jobs on a lane stop that lane
for diagnosis. All planned conditions finishing early also releases the GPU.

Runtime configuration is supplied as an untracked JSON file on the execution
host. It contains local model/data paths, GPU assignments and UUIDs. The source
repository contains portable code, candidate proposals and this study plan.
