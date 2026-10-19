"""Opt-in, portable training-state checkpoints at completed training boundaries.

These snapshots are separate from ``save_model`` exports. Callers construct a
compatible trainer (including datasets and callbacks), then resume its loop with
``train(resume_from_checkpoint=...)``. No executable objects are unpickled.
"""

from contextlib import contextmanager
from dataclasses import fields, is_dataclass
import importlib
import inspect
import hashlib
import json
import numbers
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
import uuid

import numpy as np
import torch

from comlrl.utils.distributed import unwrap_model


CHECKPOINT_VERSION = 1
STATE_FILENAME = "training_state.pt"
_OWNER_FILENAME = "checkpoint_owner.json"
_DATACLASSES = {
    "comlrl.trainers.reinforce.magrpo.NodeSample",
    "comlrl.trainers.actor_critic.iac.RolloutSample",
    "comlrl.trainers.pbrl.madpo.PreferencePair",
    "comlrl.trainers.pbrl.madpo.AgentPreferenceTensors",
    "comlrl.trainers.pbrl.iterative.PreferenceReplayShard",
}
_MODEL_FIELDS = ("agents", "critics", "reference_models", "reward_model")
_OPTIMIZER_FIELDS = (
    "optimizers",
    "agent_optimizers",
    "critic_optimizers",
    "critic_optimizer",
    "reward_optimizer",
)
_SCHEDULER_FIELDS = ("lr_scheduler", "lr_schedulers", "scheduler", "schedulers")
_RUNTIME_FIELDS = (
    "env_step",
    "_last_train_log_step",
    "rollout_buffers",
    "_comparator_rng_states",
    "_comparator_rng_streams",
    "_comparator_seed",
    "_reward_model_active",
    "_evaluating_with_task_reward",
    "_reward_distribution_range",
)
_CALLBACK_FIELDS = (
    "train_dataset",
    "reward_func",
    "reward_processor",
    "external_transition",
    "metrics_callback",
    "eval_dataset",
    "eval_logger",
    "eval_aggregator",
)
_PORTABLE_CONFIG_FIELDS = {
    "num_train_epochs",
    "num_iterations",
    "agent_devices",
    "critic_devices",
    "reference_devices",
    "comparator_devices",
    "parallel_training",
    "reward_model_device",
    "reference_device",
    "logging_steps",
    "logging_interval",
    "verbose",
}


def read_checkpoint_metadata(checkpoint):
    """Read resume metadata before constructing a trainer, without restoring state.

    ``checkpoint`` accepts a checkpoint directory or its ``training_state.pt``.
    The returned dictionary contains ``version``, ``trainer_class``, ``sequence``,
    ``run_id``, and ``wandb``. The latter is either ``None`` or the
    saved W&B ``id`` and optional ``project``/``entity``. Callers can pass that
    identity with ``resume="allow"`` in the trainer's ``wandb_config`` to avoid
    opening a new run during construction.

    Tensor storage is memory-mapped on CPU when supported by PyTorch. This does
    not initialize logging or validate compatibility with a new trainer.
    Older checkpoints may have no run identity or W&B metadata.
    """
    path = Path(checkpoint).expanduser()
    if path.is_dir():
        path = path / STATE_FILENAME
    load_options = {"map_location": "cpu", "weights_only": True}
    if "mmap" in inspect.signature(torch.load).parameters:
        load_options["mmap"] = True
    encoded = torch.load(path, **load_options)
    if (
        not isinstance(encoded, dict)
        or encoded.get("__kind__") != "dict"
        or not isinstance(encoded.get("items"), list)
    ):
        raise ValueError("Invalid training checkpoint metadata schema.")
    names = {"version", "trainer_class", "sequence", "run_id", "wandb"}
    metadata = {}
    for item in encoded["items"]:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("Invalid training checkpoint metadata schema.")
        key, value = item
        if isinstance(key, str) and key in names:
            if key in metadata:
                raise ValueError(f"Duplicate checkpoint metadata field: {key}")
            # Do not use the training-state decoder: progress and other payloads
            # may contain training records, and metadata needs only primitives.
            if key == "wandb" and value is not None:
                if (
                    not isinstance(value, dict)
                    or value.get("__kind__") != "dict"
                    or not isinstance(value.get("items"), list)
                ):
                    raise ValueError("Invalid checkpoint W&B run identity.")
                identity = {}
                for pair in value["items"]:
                    if (
                        not isinstance(pair, (list, tuple))
                        or len(pair) != 2
                        or not all(isinstance(part, str) for part in pair)
                        or pair[0] in identity
                    ):
                        raise ValueError("Invalid checkpoint W&B run identity.")
                    identity[pair[0]] = pair[1]
                value = identity
            metadata[key] = value
    if not (names - {"run_id", "wandb"}).issubset(metadata):
        raise ValueError("Incomplete training checkpoint metadata.")
    if (
        type(metadata["version"]) is not int
        or metadata["version"] != CHECKPOINT_VERSION
    ):
        raise ValueError(f"Unsupported checkpoint version: {metadata['version']}")
    if not isinstance(metadata["trainer_class"], str) or not metadata["trainer_class"]:
        raise ValueError("Invalid checkpoint trainer class.")
    if type(metadata["sequence"]) is not int or metadata["sequence"] < 0:
        raise ValueError("Invalid checkpoint sequence.")
    run_id = metadata.setdefault("run_id", None)
    if run_id is not None and (
        not isinstance(run_id, str) or re.fullmatch(r"[0-9a-f]{32}", run_id) is None
    ):
        raise ValueError("Invalid checkpoint run identity.")
    identity = metadata.setdefault("wandb", None)
    if identity is not None and (
        not isinstance(identity, dict)
        or not isinstance(identity.get("id"), str)
        or not identity["id"]
        or any(key not in {"id", "project", "entity"} for key in identity)
        or any(not isinstance(value, str) for value in identity.values())
    ):
        raise ValueError("Invalid checkpoint W&B run identity.")
    return metadata


def _class_name(value):
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _encode(value, *, copy_tensors=True):
    """Copy tensors to CPU and encode known records using safe built-in types."""
    if value is None or type(value) in (bool, int, float, str, bytes):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=copy_tensors)
    if isinstance(value, np.generic):
        return _encode(value.item(), copy_tensors=copy_tensors)
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays cannot be stored in a training checkpoint.")
        return {
            "__kind__": "numpy",
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "data": value.tobytes(),
        }
    if isinstance(value, (torch.device, torch.dtype, Path)):
        return {"__kind__": "text", "value": str(value)}
    if is_dataclass(value) and not isinstance(value, type):
        name = _class_name(value)
        if name not in _DATACLASSES:
            raise TypeError(f"Unsupported checkpoint record: {name}")
        return {
            "__kind__": "record",
            "class": name,
            "fields": {
                field.name: _encode(
                    getattr(value, field.name), copy_tensors=copy_tensors
                )
                for field in fields(value)
            },
        }
    if isinstance(value, dict):
        return {
            "__kind__": "dict",
            "items": [
                (
                    _encode(key, copy_tensors=copy_tensors),
                    _encode(item, copy_tensors=copy_tensors),
                )
                for key, item in value.items()
            ],
        }
    if isinstance(value, (list, tuple)):
        return {
            "__kind__": "tuple" if isinstance(value, tuple) else "list",
            "items": [_encode(item, copy_tensors=copy_tensors) for item in value],
        }
    raise TypeError(f"Unsupported checkpoint value: {_class_name(value)}")


def _decode(value):
    if not isinstance(value, dict):
        return value
    kind = value.get("__kind__")
    if kind == "dict":
        return {_decode(key): _decode(item) for key, item in value["items"]}
    if kind in ("tuple", "list"):
        result = [_decode(item) for item in value["items"]]
        return tuple(result) if kind == "tuple" else result
    if kind == "text":
        return value["value"]
    if kind == "numpy":
        dtype = np.dtype(value["dtype"])
        if dtype.hasobject:
            raise ValueError("Object arrays are not valid checkpoint values.")
        return np.frombuffer(value["data"], dtype=dtype).reshape(value["shape"]).copy()
    if kind == "record":
        name = value["class"]
        if name not in _DATACLASSES:
            raise ValueError(f"Unsupported checkpoint record: {name}")
        module, _, attr = name.rpartition(".")
        cls = getattr(importlib.import_module(module), attr)
        return cls(**{key: _decode(item) for key, item in value["fields"].items()})
    raise ValueError("Invalid checkpoint value encoding.")


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from the checkpoint RNG state.")
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())


def _config(trainer):
    args = getattr(trainer, "args", None)
    if args is None:
        return {}
    values = (
        {field.name: getattr(args, field.name) for field in fields(args)}
        if is_dataclass(args)
        else vars(args)
    )
    return {
        key: _decode(_encode(value))
        for key, value in values.items()
        if not key.startswith("_")
        and key not in _PORTABLE_CONFIG_FIELDS
        and not key.endswith("_dir")
    }


def _dataset_info(trainer):
    dataset = getattr(trainer, "train_dataset", None)
    if dataset is None:
        return None
    try:
        length = len(dataset)
    except TypeError:
        length = None
    return {
        "class": _class_name(dataset),
        "length": length,
        "fingerprint": getattr(dataset, "_fingerprint", None),
    }


def _fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


def _model_metadata(model, config_override=None):
    model = unwrap_model(model)
    config_model = getattr(model, "model", model)
    metadata = (
        {
            "requires_grad": {
                name: parameter.requires_grad
                for name, parameter in model.named_parameters()
            }
        }
        if isinstance(model, torch.nn.Module)
        else {}
    )
    for name in ("config", "generation_config"):
        config = getattr(model, name, getattr(config_model, name, None))
        if callable(getattr(config, "to_dict", None)):
            values = config.to_dict()
            if (
                name == "config"
                and config_override is not None
                and config is config_override[0]
            ):
                values = {**values, **config_override[1]}
            for transient in (
                "_name_or_path",
                "transformers_version",
                "_commit_hash",
                "_from_model_config",
            ):
                values.pop(transient, None)
            metadata[name] = _fingerprint(values)
    return metadata


def _tokenizer_metadata(trainer):
    result = []
    for tokenizer in getattr(trainer, "tokenizers", None) or []:
        info = {"class": _class_name(tokenizer)}
        if callable(getattr(tokenizer, "get_vocab", None)):
            info["vocab"] = _fingerprint(tokenizer.get_vocab())
        for name in (
            "special_tokens_map",
            "chat_template",
            "padding_side",
            "truncation_side",
            "model_max_length",
        ):
            if hasattr(tokenizer, name):
                info[name] = _fingerprint(getattr(tokenizer, name))
        result.append(info)
    return result


def _callback_objects(trainer):
    objects = {name: getattr(trainer, name, None) for name in _CALLBACK_FIELDS}
    for index, formatter in enumerate(getattr(trainer, "formatters", None) or []):
        objects[f"formatters.{index}"] = getattr(
            formatter, "_comlrl_checkpoint_source", formatter
        )
    return {
        name: obj
        for name, obj in objects.items()
        if callable(getattr(obj, "state_dict", None))
    }


def _components(trainer, names):
    result = {}
    for name in names:
        value = getattr(trainer, name, None)
        if value is None:
            continue
        many = isinstance(value, (list, tuple))
        items = list(value) if many else [value]
        if not all(hasattr(unwrap_model(item), "state_dict") for item in items):
            continue
        result[name] = (many, items)
    return result


def _component_states(trainer, names):
    return {
        name: {
            "many": many,
            "classes": [_class_name(unwrap_model(obj)) for obj in items],
            "states": [unwrap_model(obj).state_dict() for obj in items],
            "metadata": [_model_metadata(obj) for obj in items],
            "training": [
                (
                    {
                        key: module.training
                        for key, module in unwrap_model(obj).named_modules()
                    }
                    if isinstance(unwrap_model(obj), torch.nn.Module)
                    else None
                )
                for obj in items
            ],
        }
        for name, (many, items) in _components(trainer, names).items()
    }


def _rng_model_devices(trainer):
    override = getattr(trainer, "_checkpoint_rng_devices_override", None)
    if override is not None:
        if not isinstance(override, dict):
            raise ValueError("Invalid checkpoint RNG device override.")
        return dict(override)
    devices = {}
    for name, (_, items) in _components(
        trainer, ("agents", "critics", "reward_model")
    ).items():
        for index, item in enumerate(items):
            parameters = getattr(unwrap_model(item), "parameters", None)
            parameter = next(parameters(), None) if callable(parameters) else None
            if parameter is not None:
                devices[f"{name}.{index}"] = str(parameter.device)
    return devices


def _cuda_rng_routes(trainer, state, additional_devices=None):
    placements = state.get("rng_devices")
    current = _rng_model_devices(trainer)
    if placements is not None:
        if not isinstance(placements, dict):
            raise ValueError("Invalid checkpoint RNG device metadata.")
        if "reward_model.0" in placements and "reward_model.0" not in current:
            device = getattr(trainer, "reward_device", None)
            if device is None:
                device = getattr(
                    getattr(trainer, "args", None), "reward_model_device", None
                )
            if device is None:
                device = next(iter(current.values()), "cpu")
            current["reward_model.0"] = str(device)
    groups = [(placements, current)]
    if additional_devices is not None:
        if not isinstance(additional_devices, (list, tuple)):
            raise ValueError("Invalid nested checkpoint RNG device metadata.")
        for record in additional_devices:
            if not isinstance(record, dict) or set(record) != {"saved", "current"}:
                raise ValueError("Invalid nested checkpoint RNG device metadata.")
            groups.append((record["saved"], record["current"]))
    device_pairs = []
    for saved, live in groups:
        if not isinstance(live, dict) or (
            saved is not None and not isinstance(saved, dict)
        ):
            raise ValueError("Invalid checkpoint RNG device metadata.")
        if saved is None:
            continue
        if set(saved) != set(live):
            raise ValueError("Checkpoint RNG model slots do not match.")
        for name, previous in saved.items():
            try:
                source, destination = torch.device(previous), torch.device(live[name])
            except (TypeError, RuntimeError) as error:
                raise ValueError("Invalid checkpoint RNG device.") from error
            if source.type != destination.type and destination.type != "meta":
                raise ValueError(
                    "Cannot resume model RNG across device backends; load model weights to start a new run."
                )
            if (
                destination.type == "cuda"
                and destination.index is None
                and torch.cuda.is_available()
            ):
                destination = torch.device("cuda", torch.cuda.current_device())
            device_pairs.append((source, destination))
    if "cuda" not in state["rng"] or not torch.cuda.is_available():
        return None
    old_count = len(state["rng"]["cuda"])
    new_count = torch.cuda.device_count()
    if any(saved is None for saved, _ in groups):
        if old_count != new_count:
            raise ValueError(
                "Legacy checkpoint CUDA device count differs; save a new checkpoint before remapping devices."
            )
    routes, destinations = {}, {}
    for source, destination in device_pairs:
        if source.type != "cuda" or destination.type != "cuda":
            continue
        before, after = source.index, destination.index
        if (
            before is None
            or not 0 <= before < old_count
            or after is None
            or not 0 <= after < new_count
        ):
            raise ValueError("Checkpoint CUDA RNG device index is invalid.")
        if (after in routes and routes[after] != before) or (
            before in destinations and destinations[before] != after
        ):
            raise ValueError(
                "Cannot merge or split actor/critic CUDA RNG streams during exact resume."
            )
        routes[after], destinations[before] = before, after
    # Retain unrelated device streams when their original index still exists.
    for index in range(min(old_count, new_count)):
        routes.setdefault(index, index)
    return routes


def _restore_trainer_rng(trainer, state, additional_devices=None):
    routes = _cuda_rng_routes(trainer, state, additional_devices)
    rng = dict(state["rng"])
    if routes is not None:
        current = torch.cuda.get_rng_state_all()
        rng["cuda"] = [
            state["rng"]["cuda"][routes[index]] if index in routes else value
            for index, value in enumerate(current)
        ]
    restore_rng_state(rng)


def capture_trainer_state(trainer):
    """Capture a snapshot; nested trainers may use this without writing a file."""
    callbacks = {}
    for name, obj in _callback_objects(trainer).items():
        if not callable(getattr(obj, "load_state_dict", None)):
            raise ValueError(f"{name} provides state_dict but no load_state_dict.")
        callbacks[name] = {"class": _class_name(obj), "state": obj.state_dict()}
    extra_hook = getattr(trainer, "_checkpoint_extra_state", None)
    state = {
        "version": CHECKPOINT_VERSION,
        "trainer_class": _class_name(trainer),
        "config": _config(trainer),
        "dataset": _dataset_info(trainer),
        "tokenizers": _tokenizer_metadata(trainer),
        "models": _component_states(trainer, _MODEL_FIELDS),
        "optimizers": _component_states(trainer, _OPTIMIZER_FIELDS),
        "schedulers": _component_states(trainer, _SCHEDULER_FIELDS),
        "runtime": {
            name: getattr(trainer, name)
            for name in _RUNTIME_FIELDS
            if hasattr(trainer, name)
        },
        "callbacks": callbacks,
        "progress": getattr(trainer, "_checkpoint_progress", {}),
        "sequence": getattr(trainer, "_checkpoint_sequence", 0),
        "run_id": getattr(trainer, "_checkpoint_run_id", None),
        "wandb": getattr(trainer, "_checkpoint_wandb_identity", None),
        "rng": capture_rng_state(),
        "rng_devices": _rng_model_devices(trainer),
        "extra": extra_hook() if callable(extra_hook) else None,
    }
    # In-memory snapshots must not alias live model/optimizer tensors either.
    return _decode(_encode(state))


def _validate_component_states(
    trainer,
    saved,
    names,
    *,
    models=False,
    allow_reward=False,
    allow_absent_reward_optimizer=False,
    model_config_updates=None,
):
    if not isinstance(saved, dict):
        raise ValueError("Invalid checkpoint component schema.")
    current = _components(trainer, names)
    expected = dict(saved)
    if allow_absent_reward_optimizer and "reward_optimizer" not in expected:
        current.pop("reward_optimizer", None)
    if allow_reward:
        expected.pop("reward_model", None)
        expected.pop("reward_optimizer", None)
        current.pop("reward_model", None)
        current.pop("reward_optimizer", None)
    if set(expected) != set(current):
        raise ValueError(
            f"Checkpoint components do not match: {sorted(expected)} vs {sorted(current)}"
        )
    pending_config_updates = {}
    for name, record in expected.items():
        many, items = current[name]
        if not isinstance(record, dict) or not {
            "many",
            "states",
            "classes",
            "training",
            "metadata",
        }.issubset(record):
            raise ValueError(f"Incomplete checkpoint {name} schema.")
        if (
            record["many"] != many
            or len(record["states"]) != len(items)
            or len(record["classes"]) != len(items)
            or len(record["training"]) != len(items)
            or len(record["metadata"]) != len(items)
        ):
            raise ValueError(f"Checkpoint {name} count does not match the trainer.")
        for obj, saved_class, state in zip(items, record["classes"], record["states"]):
            obj = unwrap_model(obj)
            if not isinstance(state, dict):
                raise ValueError(f"Invalid checkpoint {name} state.")
            if _class_name(obj) != saved_class:
                raise ValueError(f"Checkpoint {name} class does not match the trainer.")
            if models:
                target = obj.state_dict()
                if set(target) != set(state):
                    raise ValueError(f"Checkpoint {name} parameter names do not match.")
                for key, tensor in target.items():
                    other = state[key]
                    if isinstance(tensor, torch.Tensor) and (
                        not isinstance(other, torch.Tensor)
                        or tensor.shape != other.shape
                        or tensor.dtype != other.dtype
                    ):
                        raise ValueError(
                            f"Checkpoint {name}.{key} shape or dtype does not match."
                        )
            elif isinstance(obj, torch.optim.Optimizer):
                if not {"state", "param_groups"}.issubset(state):
                    raise ValueError(f"Incomplete checkpoint {name} optimizer state.")
                groups = obj.state_dict()["param_groups"]
                if len(groups) != len(state.get("param_groups", [])) or any(
                    len(a["params"]) != len(b["params"])
                    for a, b in zip(groups, state["param_groups"])
                ):
                    raise ValueError(
                        f"Checkpoint {name} parameter groups do not match."
                    )
                for live_group, saved_group in zip(
                    obj.param_groups, state["param_groups"]
                ):
                    for param, param_id in zip(
                        live_group["params"], saved_group["params"]
                    ):
                        for key, value in (
                            state.get("state", {}).get(param_id, {}).items()
                        ):
                            if (
                                isinstance(value, torch.Tensor)
                                and key != "step"
                                and value.shape != param.shape
                            ):
                                raise ValueError(
                                    f"Checkpoint {name} optimizer tensor {key} has incompatible shape."
                                )
        if models:
            for index, (obj, flags, metadata) in enumerate(
                zip(items, record["training"], record["metadata"])
            ):
                if metadata != _model_metadata(obj):
                    # Some trainers apply tokenizer IDs lazily during generation.
                    # Accept only that exact transition, without changing live
                    # models until every checkpoint preflight has succeeded.
                    hook = getattr(trainer, "_checkpoint_model_config_override", None)
                    override = hook(name, index) if callable(hook) else None
                    if override is None or metadata != _model_metadata(obj, override):
                        raise ValueError(
                            f"Checkpoint {name} model or generation configuration does not match."
                        )
                    pending_config_updates[id(override[0])] = override
                modules = dict(unwrap_model(obj).named_modules())
                if (
                    not isinstance(flags, dict)
                    or set(flags) != set(modules)
                    or any(type(flag) is not bool for flag in flags.values())
                ):
                    raise ValueError(f"Invalid checkpoint {name} training modes.")

    if pending_config_updates:
        # Caller-provided actors may share a config object. A valid update for
        # one actor must not invalidate another actor or a frozen reference.
        for name, record in expected.items():
            for obj, metadata in zip(current[name][1], record["metadata"]):
                model = unwrap_model(obj)
                config = getattr(
                    model,
                    "config",
                    getattr(getattr(model, "model", model), "config", None),
                )
                override = pending_config_updates.get(id(config))
                if override is not None and metadata != _model_metadata(obj, override):
                    raise ValueError(
                        f"Checkpoint {name} has incompatible shared model configuration."
                    )
        if model_config_updates is not None:
            model_config_updates.update(pending_config_updates)


def _validate_rng(state):
    if not isinstance(state, dict) or not {"python", "numpy", "torch"}.issubset(state):
        raise ValueError("Incomplete checkpoint RNG state.")
    try:
        random.Random().setstate(state["python"])
        np.random.RandomState().set_state(state["numpy"])
        torch.Generator(device="cpu").set_state(state["torch"].cpu())
    except (TypeError, ValueError, RuntimeError, AttributeError, KeyError) as error:
        raise ValueError("Invalid checkpoint RNG state.") from error
    for tensor in list(state.get("cuda", [])) + (
        [state["mps"]] if "mps" in state else []
    ):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.uint8
            or tensor.ndim != 1
        ):
            raise ValueError("Invalid checkpoint device RNG state.")


def _validate_state(trainer, state, *, model_config_updates=None):
    required = {
        "version",
        "trainer_class",
        "config",
        "dataset",
        "models",
        "optimizers",
        "schedulers",
        "runtime",
        "callbacks",
        "progress",
        "sequence",
        "rng",
        "extra",
        "tokenizers",
    }
    if not isinstance(state, dict) or not required.issubset(state):
        raise ValueError("Incomplete training checkpoint schema.")
    if state["version"] != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint version: {state['version']}")
    if state["trainer_class"] != _class_name(trainer):
        raise ValueError("Checkpoint trainer class does not match this trainer.")
    if not isinstance(state["config"], dict):
        raise ValueError("Invalid checkpoint configuration schema.")
    saved_config = {
        key: value
        for key, value in state["config"].items()
        if key not in _PORTABLE_CONFIG_FIELDS
    }
    if saved_config != _config(trainer):
        changed = sorted(
            key
            for key in set(saved_config) | set(_config(trainer))
            if saved_config.get(key) != _config(trainer).get(key)
        )
        raise ValueError(
            f"Checkpoint configuration does not match: {', '.join(changed)}"
        )
    if state["dataset"] != _dataset_info(trainer):
        raise ValueError("Checkpoint training dataset does not match.")
    if state["tokenizers"] != _tokenizer_metadata(trainer):
        raise ValueError("Checkpoint tokenizers do not match.")
    if not isinstance(state["progress"], dict):
        raise ValueError("Invalid checkpoint progress.")
    if type(state["sequence"]) is not int or state["sequence"] < 0:
        raise ValueError("Invalid checkpoint sequence.")
    if state.get("run_id") is not None and (
        not isinstance(state["run_id"], str)
        or re.fullmatch(r"[0-9a-f]{32}", state["run_id"]) is None
    ):
        raise ValueError("Invalid checkpoint run identity.")
    identity = state.get("wandb")
    if identity is not None and (
        not isinstance(identity, dict)
        or not isinstance(identity.get("id"), str)
        or not identity["id"]
        or any(key not in {"id", "project", "entity"} for key in identity)
        or any(not isinstance(value, str) for value in identity.values())
    ):
        raise ValueError("Invalid checkpoint W&B run identity.")
    _validate_rng(state["rng"])
    if not isinstance(state["runtime"], dict) or set(state["runtime"]) - set(
        _RUNTIME_FIELDS
    ):
        raise ValueError("Unknown checkpoint runtime field.")
    comparator_validator = getattr(trainer, "_checkpoint_validate_comparator_rng", None)
    if callable(comparator_validator):
        comparator_validator(
            state["runtime"].get("_comparator_rng_states", {}),
            state["runtime"].get("_comparator_rng_streams"),
        )
        extra = state.get("extra")
        if isinstance(extra, dict):
            for name in ("_comparator_rng_states", "_comparator_rng_streams"):
                if name not in extra:
                    continue
                left, right = state["runtime"].get(name), extra[name]
                if name == "_comparator_rng_states":
                    matching = (
                        isinstance(left, dict)
                        and isinstance(right, dict)
                        and set(left) == set(right)
                        and all(
                            isinstance(right[key], torch.Tensor)
                            and torch.equal(value, right[key])
                            for key, value in left.items()
                        )
                    )
                else:
                    matching = left == right
                if not matching:
                    raise ValueError(
                        "Checkpoint comparator runtime and extra state disagree."
                    )
    progress_validator = getattr(trainer, "_checkpoint_validate_progress", None)
    if callable(progress_validator):
        progress_validator(state)
    if not isinstance(state["callbacks"], dict):
        raise ValueError("Invalid checkpoint callback schema.")
    _cuda_rng_routes(trainer, state)
    _validate_wandb_resume(trainer, state.get("wandb"))
    reward_init = (
        "reward_model" in state["models"]
        and getattr(trainer, "reward_model", None) is None
    )
    if reward_init and not callable(getattr(trainer, "_init_reward_model", None)):
        raise ValueError(
            "Checkpoint requires a reward model that cannot be initialized."
        )
    _validate_component_states(
        trainer,
        state["models"],
        _MODEL_FIELDS,
        models=True,
        allow_reward=reward_init,
        model_config_updates=model_config_updates,
    )
    _validate_component_states(
        trainer,
        state["optimizers"],
        _OPTIMIZER_FIELDS,
        allow_reward=reward_init,
        allow_absent_reward_optimizer="reward_model" in state["models"],
    )
    _validate_component_states(trainer, state["schedulers"], _SCHEDULER_FIELDS)
    current_callbacks = _callback_objects(trainer)
    if set(state["callbacks"]) != set(current_callbacks):
        raise ValueError("Checkpoint stateful callbacks do not match.")
    for name, record in state["callbacks"].items():
        if not isinstance(record, dict) or not {"class", "state"}.issubset(record):
            raise ValueError(f"Incomplete checkpoint callback {name} schema.")
        callback = current_callbacks[name]
        if _class_name(callback) != record["class"] or not callable(
            getattr(callback, "load_state_dict", None)
        ):
            raise ValueError(f"Checkpoint callback {name} does not match.")
    return reward_init


def restore_trainer_state(trainer, state, checkpoint_dir=None, restore_rng=True):
    """Validate compatibility, then restore an in-memory snapshot."""
    model_config_updates = {}
    reward_init = _validate_state(
        trainer, state, model_config_updates=model_config_updates
    )
    extra_validator = getattr(trainer, "_checkpoint_validate_extra_state", None)
    extra_validation = None
    if callable(extra_validator):
        extra_validation = extra_validator(state["extra"], checkpoint_dir)
    additional_devices = (
        extra_validation.get("rng_devices")
        if isinstance(extra_validation, dict)
        else None
    )
    _cuda_rng_routes(trainer, state, additional_devices)
    assets_validator = getattr(trainer, "_checkpoint_validate_assets", None)
    if callable(assets_validator) and "assets" in state:
        assets_validator(checkpoint_dir, state["assets"])
    if reward_init:
        rng = capture_rng_state()
        old_attributes = dict(vars(trainer))
        try:
            trainer._init_reward_model()
            if "reward_optimizer" not in state["optimizers"]:
                trainer.reward_optimizer = None
            _validate_component_states(
                trainer,
                state["models"],
                _MODEL_FIELDS,
                models=True,
                model_config_updates=model_config_updates,
            )
            _validate_component_states(trainer, state["optimizers"], _OPTIMIZER_FIELDS)
            # A custom lazy factory may choose a different device than its
            # configuration predicted. Reject that before loading live state.
            _cuda_rng_routes(trainer, state, additional_devices)
        except BaseException:
            vars(trainer).clear()
            vars(trainer).update(old_attributes)
            raise
        finally:
            restore_rng_state(rng)
    for config, values in model_config_updates.values():
        for name, value in values.items():
            setattr(config, name, value)
    if (
        "reward_model" in state["models"]
        and "reward_optimizer" not in state["optimizers"]
    ):
        trainer.reward_optimizer = None
    for names, key in (
        (_MODEL_FIELDS, "models"),
        (_OPTIMIZER_FIELDS, "optimizers"),
        (_SCHEDULER_FIELDS, "schedulers"),
    ):
        current = _components(trainer, names)
        for name, record in state[key].items():
            for index, (obj, saved) in enumerate(
                zip(current[name][1], record["states"])
            ):
                # Optimizer.load_state_dict may reuse CPU tensors from its input.
                # Preserve reusable snapshots when the resumed run updates them.
                restored = saved if key == "models" else _decode(_encode(saved))
                unwrap_model(obj).load_state_dict(restored)
                if isinstance(unwrap_model(obj), torch.nn.Module):
                    for module_name, module in unwrap_model(obj).named_modules():
                        module.training = record["training"][index][module_name]
    for name, value in state["runtime"].items():
        if name not in _RUNTIME_FIELDS:
            raise ValueError(f"Unknown checkpoint runtime field: {name}")
        setattr(trainer, name, _decode(_encode(value)))
    for name, record in state["callbacks"].items():
        _callback_objects(trainer)[name].load_state_dict(
            _decode(_encode(record["state"]))
        )
    trainer._checkpoint_progress = _decode(_encode(state["progress"]))
    trainer._checkpoint_sequence = state["sequence"]
    trainer._checkpoint_run_id = state.get("run_id") or uuid.uuid4().hex
    assets_hook = getattr(trainer, "_checkpoint_load_assets", None)
    if callable(assets_hook) and "assets" in state:
        assets_hook(checkpoint_dir, state["assets"])
    extra_hook = getattr(trainer, "_restore_checkpoint_extra_state", None)
    if callable(extra_hook):
        extra_hook(state["extra"], checkpoint_dir)
    _restore_wandb_identity(trainer, state.get("wandb"))
    if restore_rng:
        _restore_trainer_rng(trainer, state, additional_devices)
    trainer._checkpoint_save_blocked = False
    trainer._checkpoint_untracked_preference = False


def _fsync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root):
    """Flush every asset before publishing the directory entry."""
    for directory, _, files in os.walk(root, topdown=False):
        for name in files:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError("Checkpoint assets must not contain symbolic links.")
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        _fsync_directory(directory)


def _validate_wandb_resume(trainer, identity):
    config = getattr(trainer, "wandb_config", None)
    if (
        not identity
        or not isinstance(config, dict)
        or "id" in config
        or "resume" in config
    ):
        return
    previous = getattr(trainer, "_checkpoint_wandb_identity", None)
    if getattr(trainer, "wandb_initialized", False) and (
        not previous or previous.get("id") != identity["id"]
    ):
        if not getattr(trainer, "_checkpoint_wandb_owned", False):
            raise ValueError(
                "Cannot replace an external W&B run; finish it or specify id/resume explicitly before restoring."
            )


def _restore_wandb_identity(trainer, identity):
    config = getattr(trainer, "wandb_config", None)
    if not identity or not isinstance(config, dict):
        return
    if "id" in config or "resume" in config:
        return
    previous = getattr(trainer, "_checkpoint_wandb_identity", None)
    if getattr(trainer, "wandb_initialized", False):
        if previous and previous.get("id") == identity["id"]:
            return
        # Constructors retain their original eager logging behavior. Only finish
        # the run opened by this trainer, never an unrelated global W&B run.
        run = getattr(trainer, "_checkpoint_wandb_run", None)
        if run is None:
            raise ValueError(
                "Cannot resume W&B without the trainer's owned run handle."
            )
        run.finish()
        trainer.wandb_initialized = False
    trainer.wandb_config = dict(config)
    trainer.wandb_config.update(id=identity["id"], resume="allow")
    for key in ("project", "entity"):
        if key in identity and not trainer.wandb_config.get(key):
            trainer.wandb_config[key] = identity[key]
    trainer._checkpoint_wandb_identity = dict(identity)


def _check_checkpoint_destination(target, run_id, sequence):
    """Confirm an existing automatic snapshot belongs to the resumed run."""
    if target.is_symlink() or not target.is_dir():
        raise FileExistsError(
            f"Checkpoint destination is not a checkpoint directory: {target}"
        )
    try:
        owner = json.loads((target / _OWNER_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        owner = None
    if (
        not isinstance(owner, dict)
        or not isinstance(owner.get("run_id"), str)
        or re.fullmatch(r"[0-9a-f]{32}", owner["run_id"]) is None
        or type(owner.get("sequence")) is not int
    ):
        raise FileExistsError(
            f"Legacy or unowned checkpoint has no valid ownership metadata: {target}. "
            "Use an empty checkpoint_dir to continue."
        )
    if run_id is None or owner["run_id"] != run_id:
        raise FileExistsError(
            f"Refusing to overwrite another run's checkpoint: {target}"
        )
    if owner["sequence"] != sequence or not (target / STATE_FILENAME).is_file():
        raise FileExistsError(
            f"Checkpoint destination has inconsistent ownership metadata: {target}"
        )


def _preflight_checkpoint_directory(checkpoint_dir, metadata=None):
    """Reject predictable automatic-save collisions before restoring or training."""
    if checkpoint_dir is None:
        return
    root = Path(checkpoint_dir).expanduser().absolute()
    if root.is_symlink() and not root.exists():
        raise NotADirectoryError(f"checkpoint_dir is not a directory: {root}")
    if not root.exists():
        return
    if not root.is_dir():
        raise NotADirectoryError(f"checkpoint_dir is not a directory: {root}")
    for target in sorted(root.iterdir()):
        match = re.fullmatch(r"checkpoint-(\d+)", target.name)
        if match is None:
            continue
        if metadata is None:
            raise FileExistsError(
                f"Checkpoint directory already contains another run's checkpoint: {target}. "
                "Resume a checkpoint or use an empty checkpoint_dir."
            )
        sequence = int(match.group(1))
        if sequence > metadata["sequence"]:
            _check_checkpoint_destination(target, metadata["run_id"], sequence)


class TrainingCheckpointMixin:
    """Explicit checkpoint APIs; ordinary training performs no checkpoint I/O."""

    def _checkpoint_init_wandb(self):
        """Initialize resumed logging without advancing training RNG streams."""
        if getattr(self, "wandb_config", None) is None or getattr(
            self, "wandb_initialized", False
        ):
            return
        rng = capture_rng_state()
        try:
            self._init_wandb()
        finally:
            restore_rng_state(rng)

    def _checkpoint_record_wandb_run(self, run, previous_run=None):
        if run is None or not isinstance(getattr(run, "id", None), str):
            return
        self._checkpoint_wandb_run = run
        self._checkpoint_wandb_owned = run is not previous_run
        self._checkpoint_wandb_identity = {
            key: value
            for key in ("id", "project", "entity")
            if isinstance(value := getattr(run, key, None), str) and value
        }

    def save_checkpoint(self, output_dir):
        """Atomically save at an idle or completed training boundary."""
        if getattr(self, "_checkpoint_context_active", False) and not getattr(
            self, "_checkpoint_safe_to_save", False
        ):
            raise RuntimeError(
                "Checkpoint saving is only supported at a completed training boundary."
            )
        if getattr(self, "_checkpoint_save_blocked", False) and not getattr(
            self, "_checkpoint_safe_to_save", False
        ):
            raise RuntimeError(
                "Training did not reach a completed checkpoint boundary; resume the last saved checkpoint."
            )
        if getattr(self, "_checkpoint_untracked_preference", False):
            raise RuntimeError(
                "Preference checkpoint tracking was not enabled. Use train(checkpoint_dir=...) to retain resumable state."
            )
        target = Path(output_dir).expanduser().absolute()
        if target.is_symlink() or (
            target.exists() and not (target / STATE_FILENAME).is_file()
        ):
            raise FileExistsError(
                f"Refusing to replace a non-checkpoint directory: {target}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
        backup = None
        rng = capture_rng_state()
        try:
            assets_hook = getattr(self, "_checkpoint_save_assets", None)
            assets = assets_hook(staging) if callable(assets_hook) else None
            state = capture_trainer_state(self)
            state["rng"] = rng
            if assets is not None:
                state["assets"] = assets
            with (staging / STATE_FILENAME).open("wb") as stream:
                torch.save(_encode(state, copy_tensors=False), stream)
                stream.flush()
            if state.get("run_id") is not None:
                (staging / _OWNER_FILENAME).write_text(
                    json.dumps(
                        {"run_id": state["run_id"], "sequence": state["sequence"]}
                    ),
                    encoding="utf-8",
                )
            _fsync_tree(staging)
            if target.exists():
                backup = Path(
                    tempfile.mkdtemp(prefix=f".{target.name}-old-", dir=target.parent)
                )
                backup.rmdir()
                os.replace(target, backup)
            try:
                os.replace(staging, target)
            except BaseException:
                if backup is not None:
                    os.replace(backup, target)
                    backup = None
                raise
            _fsync_directory(target.parent)
            if backup is not None:
                shutil.rmtree(backup)
                _fsync_directory(target.parent)
            return str(target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
            restore_rng_state(rng)

    def load_checkpoint(self, checkpoint):
        """Restore state into a compatible trainer; ``train(resume_from_checkpoint=...)``
        additionally resumes its saved loop position.
        """
        path = Path(checkpoint).expanduser()
        if path.is_dir():
            path = path / STATE_FILENAME
        state = _decode(torch.load(path, map_location="cpu", weights_only=True))
        restore_trainer_state(self, state, checkpoint_dir=path.parent)
        return self._checkpoint_progress

    @contextmanager
    def _checkpoint_context(
        self,
        *,
        resume_from_checkpoint=None,
        checkpoint_dir=None,
        checkpoint_interval=1,
        checkpoint_total_limit=None,
    ):
        if (
            isinstance(checkpoint_interval, bool)
            or not isinstance(checkpoint_interval, numbers.Integral)
            or checkpoint_interval < 1
        ):
            raise ValueError("checkpoint_interval must be a positive integer.")
        if checkpoint_total_limit is not None and (
            isinstance(checkpoint_total_limit, bool)
            or not isinstance(checkpoint_total_limit, numbers.Integral)
            or checkpoint_total_limit < 1
        ):
            raise ValueError(
                "checkpoint_total_limit must be a positive integer or None."
            )
        if getattr(self, "_checkpoint_context_active", False):
            raise RuntimeError(
                "Nested checkpoint contexts require an explicit progress callback."
            )
        metadata = (
            read_checkpoint_metadata(resume_from_checkpoint)
            if resume_from_checkpoint is not None and checkpoint_dir is not None
            else None
        )
        _preflight_checkpoint_directory(checkpoint_dir, metadata)
        self._checkpoint_directory = checkpoint_dir
        if resume_from_checkpoint is not None:
            self.load_checkpoint(resume_from_checkpoint)
        else:
            self._checkpoint_progress = {}
            self._checkpoint_sequence = 0
            self._checkpoint_run_id = uuid.uuid4().hex
        prepare = getattr(self, "_checkpoint_prepare_training", None)
        if callable(prepare):
            prepare()
        self._checkpoint_interval = int(checkpoint_interval)
        self._checkpoint_total_limit = checkpoint_total_limit
        self._checkpoint_context_active = True
        self._checkpoint_safe_to_save = False
        self._checkpoint_save_blocked = True
        self._checkpoint_untracked_preference = False
        try:
            yield self._checkpoint_progress
        except BaseException:
            self._checkpoint_save_blocked = True
            raise
        else:
            self._checkpoint_save_blocked = False
        finally:
            self._checkpoint_context_active = False

    def _checkpoint_commit(self, progress, force=False):
        self._checkpoint_progress = progress
        self._checkpoint_sequence = getattr(self, "_checkpoint_sequence", 0) + 1
        directory = getattr(self, "_checkpoint_directory", None)
        if directory is not None and (
            force or self._checkpoint_sequence % self._checkpoint_interval == 0
        ):
            self._checkpoint_safe_to_save = True
            try:
                target = (
                    Path(directory).expanduser().absolute()
                    / f"checkpoint-{self._checkpoint_sequence}"
                )
                if target.exists() or target.is_symlink():
                    _check_checkpoint_destination(
                        target, self._checkpoint_run_id, self._checkpoint_sequence
                    )
                self.save_checkpoint(target)
                self._checkpoint_prune(target)
            finally:
                self._checkpoint_safe_to_save = False

    def _checkpoint_prune(self, latest):
        limit = getattr(self, "_checkpoint_total_limit", None)
        if limit is None:
            return
        root = Path(latest).parent
        checkpoints = []
        for path in root.iterdir():
            if (
                path.is_symlink()
                or not path.is_dir()
                or not re.fullmatch(r"checkpoint-\d+", path.name)
            ):
                continue
            try:
                owner = json.loads((path / _OWNER_FILENAME).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if (
                isinstance(owner, dict)
                and owner.get("run_id") == self._checkpoint_run_id
                and type(owner.get("sequence")) is int
                and (path / STATE_FILENAME).is_file()
            ):
                checkpoints.append((owner["sequence"], path))
        # Always retain the newly committed snapshot, including when branching
        # from an older checkpoint in a directory containing later snapshots.
        older = sorted(
            (item for item in checkpoints if item[1] != latest), reverse=True
        )
        for _, path in older[max(0, int(limit) - 1) :]:
            shutil.rmtree(path)
        _fsync_directory(root)
        if callable(getattr(self, "_checkpoint_save_assets", None)):
            from comlrl.trainers.pbrl.checkpoint_extras import (
                prune_preference_asset_pool,
            )

            prune_preference_asset_pool(root)
