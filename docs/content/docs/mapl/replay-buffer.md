---
title: Replay Buffer
weight: 3
math: true
---

Iterative preference training collects a new set of comparison pairs each round. The replay buffer decides which collected pairs are used for that round's updates. It is shared by `MADPOIterTrainer` and `MARLHFIterTrainer`.

The current round is written before replay selection, so it is eligible alongside earlier rounds. Replay controls the training data; the [comparator policy]({{< relref "/docs/mapl/comparator" >}}) controls how new candidate responses are generated.

## Replay Modes

| Mode | Eligible preference data |
| --- | --- |
| `current` | All pairs from the current round, without historical sampling. |
| `nearest_k` | The latest `preference_replay_k` non-empty iteration datasets, including the current one when non-empty. |
| `all_history` | Every recorded non-empty iteration dataset in the current trainer run. |
| `lambda_decay` | The same history, with greater sampling weight on more recent datasets when λ is below 1. |

`nearest_k` means recent iterations in time. It does not perform nearest-neighbor retrieval by embedding or similarity.

`all_history` makes all recorded iterations eligible; it does not necessarily train on every historical pair in each round. For `nearest_k` and `all_history`, eligible iteration datasets have equal selection weight, regardless of how many pairs each contains.

## Lambda Decay

`preference_replay_lambda` controls how quickly the probability of selecting an older dataset decreases. If age 0 denotes the newest non-empty dataset and there are {{< katex >}}K{{< /katex >}} eligible datasets, the normalized weight for age {{< katex >}}a{{< /katex >}} is:

{{< katex display=true >}}
w_a = \frac{(1-\lambda)\lambda^a}{\sum_{j=0}^{K-1}(1-\lambda)\lambda^j},
\qquad 0 \leq \lambda < 1.
{{< /katex >}}

- `lambda=0` selects only the newest non-empty dataset.
- Values between 0 and 1 assign progressively less weight to older datasets.
- `lambda=1` is handled separately and gives every eligible dataset equal weight.

The weights apply to iteration datasets, then pairs are sampled within each selected dataset. A smaller λ emphasizes recent experience while a larger λ keeps more earlier data in the sampling distribution.

## Sampling Size

Historical replay first samples dataset assignments, then samples pairs from each selected dataset. Within one dataset, sampling uses no repeated pairs when its assigned count fits in that dataset. Larger assigned counts cycle through shuffled copies of the dataset, so repeated pairs are possible. The combined sample is shuffled before training.

By default, `preference_replay_sample_size` uses the number of pairs collected in the current round. If that round has no pairs, it uses the size of the latest non-empty dataset. An explicit positive sample size overrides this behavior.

{{% hint info %}}
**Replay Buffer** parameters:

- `preference_replay_mode`: `"current"` (default), `"nearest_k"`, `"all_history"`, or `"lambda_decay"`
- `preference_replay_k`: Positive number of recent non-empty datasets eligible for `"nearest_k"`; required in that mode
- `preference_replay_lambda`: Age-decay parameter in [0, 1]; required only for `"lambda_decay"`
- `preference_replay_sample_size`: Positive number of pairs sampled for historical replay, or `None` for the default described above
- `preference_replay_dir`: Optional location for serialized preference datasets
{{% /hint %}}

`current` rejects `preference_replay_sample_size` and `preference_replay_lambda`; its `preference_replay_k` may be `None` or 1. For the historical modes, use `preference_replay_k` only with `nearest_k` and `preference_replay_lambda` only with `lambda_decay`.

## Configuration Examples

Recent iterations:

```python
from comlrl.trainers.preference import MADPOIterConfig

args = MADPOIterConfig(
    num_agents=2,
    preference_replay_mode="nearest_k",
    preference_replay_k=3,
    preference_replay_sample_size=256,
)
```

Age-weighted history:

```python
from comlrl.trainers.preference import MADPOIterConfig

args = MADPOIterConfig(
    num_agents=2,
    preference_replay_mode="lambda_decay",
    preference_replay_lambda=0.8,
    preference_replay_sample_size=256,
)
```

The same replay options are available in `MARLHFIterConfig`. Both examples configure sampling only; model sources, datasets, reward functions, and the other trainer inputs are supplied separately.

## Saved Preference Data

The trainer writes each round to `iteration_0001.json`, `iteration_0002.json`, and so on under the replay directory. Records contain the prompts, preferred and rejected completions, rewards, and associated preference metadata. Historical samples are reconstructed from this text using the actor tokenizers.

The in-memory replay index tracks datasets produced by the current trainer instance. Setting `preference_replay_dir` changes the output location; it does not automatically import an earlier run's files.

Actor checkpoints are stored separately through `policy_checkpoint_dir`. They provide model weights for a `history` comparator, while the replay buffer stores preference examples used for training.
