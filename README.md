# Pop-OpenES AutoResearch baseline

## Environment

Use Python 3.12 and the uv environment in this directory:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
```

`.python-version` selects Python 3.12. The requirements pin CUDA 12 JAX 0.11.2,
stable Hydra 1.3.2 / OmegaConf 2.3.0, and Aim 3.29.1. EvoRL stays editable at
`/home/zaber/workspace/evorl`. The experiment's policy workflow and optimizer are
local (`openes.py`, `openes_optimizer.py`). The inherited default recorders are
`[log, aim]`; the Pop-OpenES config does not override them.

## Algorithm and configuration

```bash
python train.py agent=meta/pop-openes env=brax/ant num_instances=16 num_iters=1000
```

`PopOpenESWorkflow` maintains an outer population of OpenES instances, vectorized
on one GPU. Every member starts with an independent random key and retains its own
center policy, Adam state, random stream and observation normalization statistics.
All members use the same hyperparameters. Evaluation selects the best current
center; the members continue training independently. There is no PBT search space,
exploitation, parameter copying, hyperparameter perturbation or restart mechanism.

The two population sizes refer to different levels:

- `num_instances`: number of OpenES instances (default 16).
- `target_workflow.pop_size`: candidate policies per OpenES update (default 128).

`seed` is the root random seed (default 42). Member keys are generated with
`jax.random.split(jax.random.key(seed), num_instances)`. Member indices are zero-based
population positions, not integer training seeds 42, 43, etc. A new root seed
produces a new population of independent runs.

Inner OpenES settings come from `configs/agent/openes.yaml`; Pop-OpenES overrides
`num_envs=1` and `episodes_for_fitness=1`. Other defaults are Adam, centered ranks,
antithetic sampling, learning rate 0.01, perturbation scale 0.02, weight decay 0.005, a [256, 256] MLP, and 10,000 random
observation-normalization steps. Inner settings live under `target_workflow`, e.g.
`target_workflow.pop_size=256`, `target_workflow.episodes_for_fitness=8`, and
`target_workflow.ec_lr.init=0.005`.
`num_iters` counts updates per instance and defaults to 2,000; use 1,000 to match
the report's initial Ant training length. The outer `num_iters` and `eval_interval`
are independent settings. The nested fields with the same names are ignored: the
outer loop calls the inner `setup()`, `step()` and `evaluate()`, never `learn()`.

Every `eval_interval` updates (default 50), each member's current ES center policy
is evaluated over `target_workflow.eval_episodes` episodes (default 128). The member
with the highest mean episode return wins that evaluation. The final iteration is
always evaluated. Selection is over the current population at the current update.

## Metrics

The metric classes in `pop_openes.py` separate progress, training and evaluation.

| Class / field | Meaning |
| --- | --- |
| `PopOpenESWorkflowMetric.iterations` | Completed OpenES updates per member |
| `sampled_episodes` | Cumulative training episodes across all members |
| `sampled_timesteps_m` | Cumulative training transitions across all members, in millions |
| `PopOpenESTrainMetric.pop_fitnesses` | `[members, candidates]` array; each candidate's mean training return |
| `pop_ec_metrics` | Inner optimizer diagnostics with a leading member axis |
| `PopOpenESEvalMetric.pop_episode_returns` | `[members]` array of center-policy mean evaluation returns |
| `pop_episode_lengths` | `[members]` array of center-policy mean episode lengths |
| `best_member_index` | Index selected by the highest mean evaluation return |
| `best_episode_return` | Selected member's mean evaluation return |
| `best_episode_length` | The same selected member's mean evaluation length |

Each completed update writes one recorder event containing progress and training
metrics, plus evaluation metrics at evaluation updates. All series use the outer
iteration as their recorder step. Training fitness and center-policy evaluation
return have separate names. Population mean and standard deviation are derived
when logging, rather than stored as additional metric fields. Training logs include
`train/pop_fitnesses/{min,max,mean,std}` and the same statistics for each member at
`train/members/<index>/fitness`. Per-member `sampled_episodes` and
`sampled_timesteps_m` accompany those curves; root sampling counters sum all members.

Evaluation logs include:

- `eval/best_episode_return`, `eval/best_episode_length`, `eval/best_member_index`.
- `eval/pop_episode_returns/{min,max,mean,std,values}` and the same for lengths.
- `eval/members/<index>/{episode_return,episode_length}` as scalar time series.

`std` is the population standard deviation (`ddof=0`); for evaluation summaries,
this is the variation across instance means, not across individual episodes.
Non-finite training fitness or evaluation returns/lengths abort the run with the
iteration and invalid indices instead of silently omitting invalid results.
The report's across-run mean corresponds to `eval/pop_episode_returns/mean`; the Pop-OpenES selection score
is `eval/best_episode_return`. These summaries describe the evaluated centers.

Sampling counters exclude warmup and evaluation, following EvoRL. With the
current 16 instances, 128 candidates and one fitness episode, one update consumes
2,048 training episodes in total (4.096 million at 2,000 updates; 256,000 per
instance). Comparisons should keep the instance count and aggregate training
budget fixed. Both episode counts must be exact multiples of their corresponding
environment counts; OpenES asserts this during construction, before creating any
environments, so the sampling budget is not silently rounded up.

`get_best_agent_state(state)` extracts the winner's center policy with its own
observation statistics. `get_best_workflow_state(state)` includes its optimizer.
Pop-OpenES enables checkpointing by default (`checkpoint.enable=true`).
`checkpoint.save_interval_steps` controls periodic saves (default 100 updates),
independently of `eval_interval`. `learn()` calls the checkpoint manager at each
update and forces a final save. Orbax also saves the first requested step when
no checkpoint exists yet. Checkpoints contain all member states and the selected
index. The index is -1 before evaluation and after another training
update; if a checkpoint was saved between evaluations, evaluate the restored state
before extracting its best policy.

## Configuration composition

The Pop-OpenES config places the existing OpenES parameters under `target_workflow`:

```yaml
# @package _global_
defaults:
  - /agent@target_workflow: openes
  - _self_

workflow_cls: pop_openes.PopOpenESWorkflow
num_instances: 16
num_iters: 2000
eval_interval: 50
```

`/agent` selects the root `agent` config group, `openes` loads
`configs/agent/openes.yaml`, and `@target_workflow` nests the imported settings.
`_self_` merges this file's own content afterward. The resulting shape is:

```yaml
workflow_cls: pop_openes.PopOpenESWorkflow
num_instances: 16
num_iters: 2000
eval_interval: 50
target_workflow:
  workflow_cls: openes.OpenESWorkflow
  pop_size: 128
  eval_episodes: 128
  # Other OpenES settings are nested here as well.
```

`num_instances` counts complete OpenES workflows. The nested
`target_workflow.pop_size` keeps OpenES's original candidate-population meaning.
Only the outer `num_iters` and `eval_interval` control the run; changing the nested
values does not change the training length or evaluation schedule.
`PopOpenESWorkflow` imports `OpenESWorkflow` directly; it does not validate or
resolve the nested class name. A private copy of the inner config receives the
shared environment and disables inner checkpointing, while the outer checkpoint
saves the whole population.

## Launch and verify

```bash
./run.sh num_instances=16 num_iters=1000
```

This launches Ant, Hopper and Walker2d with Hydra/joblib, one GPU per job. Each
job's outer population is vectorized on that GPU. Set `CUDA_VISIBLE_DEVICES` and
`N_JOBS` to choose resources. Use `seed=42,43` for repeats of the entire population.
The original experiment is still available with `agent=openes`.

`train.py` defaults to on-demand JAX allocation by setting
`XLA_PYTHON_CLIENT_PREALLOCATE=false` before initializing JAX; an explicitly set
value is respected. This does not cap process memory. `run.sh` currently defaults
to three concurrent jobs; on a single shared GPU use `N_JOBS=1` or budget for all
concurrent jobs together.

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m unittest discover -s tests -v
```

Tests use tiny Brax Fast runs to compare vectorized training with independent
runs, verify member-specific normalization and selection, and check logging,
forced final evaluation, checkpoint restore, exact episode batches and non-finite
result handling. GPU smoke validation should also
exercise Ant through `train.py` with the default Log and Aim recorders.
