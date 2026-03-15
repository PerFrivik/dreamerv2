# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install dependencies:**
```sh
pip3 install tensorflow==2.6.0 tensorflow_probability ruamel.yaml 'gym[atari]' dm_control
```

**Train on Atari:**
```sh
python3 dreamerv2/train.py --logdir ~/logdir/atari_pong/dreamerv2/1 \
  --configs atari --task atari_pong
```

**Train on DM Control (vision):**
```sh
python3 dreamerv2/train.py --logdir ~/logdir/dmc_walker_walk/dreamerv2/1 \
  --configs dmc_vision --task dmc_walker_walk
```

**Debug mode** (disables `tf.function`, smaller batch, more frequent eval):
```sh
python3 dreamerv2/train.py --logdir ~/logdir/debug --configs atari debug --task atari_pong
```

**Monitor with TensorBoard:**
```sh
tensorboard --logdir ~/logdir
```

**Generate plots:**
```sh
python3 dreamerv2/common/plot.py --indir ~/logdir --outdir ~/plots \
  --xaxis step --yaxis eval_return --bins 1e6
```

**Use as a library (custom environments):**
```python
import dreamerv2.api as dv2
config = dv2.defaults.update({...}).parse_flags()
dv2.train(env, config)
```

## Architecture

The codebase has two entry points: `dreamerv2/train.py` (CLI, multi-env, train/eval split) and `dreamerv2/api.py` (library API for custom gym environments). Both converge on the same `agent.Agent` class.

### Core training loop (`train.py`)
The main loop alternates: collect experience → train world model → train actor-critic → evaluate. Experience is stored in `common.Replay` as episode files. `common.Driver` manages environment stepping and fires callbacks (`on_step`, `on_episode`, `on_reset`). `common.Counter` tracks global steps; `common.Logger` writes to terminal, JSONL, and TensorBoard.

### Agent (`agent.py`)
Three classes:

- **`WorldModel`**: Wraps `EnsembleRSSM` + `Encoder` + heads (`Decoder`, reward MLP, optional discount MLP). Trained end-to-end with KL + reconstruction losses. `loss_scales` in config controls relative weighting. `grad_heads` controls which heads receive gradients back into the RSSM.
- **`ActorCritic`**: Actor and critic MLPs trained on *imagined* trajectories from the world model via `WorldModel.imagine()`. Supports `dynamics` (straight-through), `reinforce`, or `both` gradient estimators for the actor (auto-selected based on discrete vs. continuous actions). Uses λ-returns and an optional slow target critic.
- **`Agent`**: Composes `WorldModel` and `ActorCritic`. `policy()` runs inference (encode → RSSM obs step → actor). `train()` runs one world model update then one actor-critic update. Checkpoints saved as `variables.pkl` via pickle.

### World model internals (`common/nets.py`)
- **`EnsembleRSSM`**: Recurrent State Space Model with a GRU deterministic state and categorical/Gaussian stochastic state. The ensemble is used only for the prior (`img_step`) — a random ensemble member is sampled each step, which provides implicit uncertainty without multiple forward passes at inference. `observe()` runs the posterior (with observations); `imagine()` runs the prior only (no observations).
- **`Encoder`/`Decoder`**: Accept regex patterns (`cnn_keys`, `mlp_keys`) to route observation keys through CNN or MLP paths. CNN uses strided convolutions; decoder uses transposed convolutions.
- **`Module.get(name, ctor, ...)`**: Lazy layer creation by name — layers are created on first call and reused on subsequent calls. This is how the RSSM builds its sub-layers without explicit `__init__` declarations.

### Configuration system (`common/config.py`, `configs.yaml`)
Configs are YAML with inheritance. `--configs atari debug` merges `defaults` → `atari` → `debug` in order, with later entries overriding earlier ones. Individual keys can be overridden via CLI flags (e.g., `--model_opt.lr 1e-3`). Dot notation accesses nested keys. Config objects are immutable; `.update()` returns a new config.

### Exploration (`expl.py`)
`expl_behavior` config selects the exploration strategy:
- `greedy`: uses the task actor (default)
- `Plan2Explore`: trains an ensemble of disagreement models as intrinsic reward
- `ModelLoss`: uses a model-based intrinsic reward

### Key conventions
- `tfutils.py` monkey-patches TensorFlow tensors with numpy-style methods (`.mean()`, `.reshape()`, `.astype()`, etc.) and adds `tf.tensor` as an alias for `tf.convert_to_tensor`.
- Mixed precision (float16) is on by default; disable with `--precision 32`.
- `@tf.function` is applied to hot paths; `debug` config disables JIT for step-by-step debugging.
- Metrics logged as scalars are averaged over the log interval; the raw values accumulate in a `defaultdict(list)` and are cleared after each log write.
