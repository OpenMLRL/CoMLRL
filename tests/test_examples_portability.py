import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

from datasets import Dataset
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
)
import wandb

from comlrl.trainers.actor_critic import IACTrainer
from comlrl.trainers.reinforce import MAGRPOTrainer


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SCRIPTS = ["tldr-len-ratio", "story-len-ratio", "leetcode-func-print"]


def _load_example(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def offline_entrypoint(monkeypatch):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"<unk>": 0, "<eos>": 1}, unk_token="<unk>")
        ),
        unk_token="<unk>",
        eos_token="<eos>",
        pad_token="<eos>",
    )
    loaded_models = []
    trained = []
    saved = []
    logged = []

    def load_model(name, **kwargs):
        loaded_models.append(name)
        return GPT2LMHeadModel(GPT2Config(vocab_size=2, n_embd=8, n_layer=1, n_head=1))

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", load_model)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(wandb, "init", lambda **kwargs: logged.append(kwargs))
    for trainer_cls in (IACTrainer, MAGRPOTrainer):
        monkeypatch.setattr(trainer_cls, "train", lambda self: trained.append(self))
        monkeypatch.setattr(
            trainer_cls, "save_model", lambda self, path: saved.append(path)
        )

    def run(script, *args):
        module = _load_example(script)
        if script == "tldr-len-ratio":
            monkeypatch.setattr(
                module,
                "load_dataset",
                lambda *a, **k: Dataset.from_dict({"prompt": ["Example prompt"]}),
            )
        monkeypatch.setattr(sys, "argv", [f"{script}.py", *args])
        entrypoint = getattr(module, "main", None) or module.example_usage
        entrypoint()
        return trained[-1]

    return SimpleNamespace(
        run=run,
        loaded_models=loaded_models,
        saved=saved,
        logged=logged,
        tokenizer=tokenizer,
    )


@pytest.mark.parametrize("script,epochs", list(zip(SCRIPTS, [15, 3, 100])))
def test_examples_reach_training_without_scheduler_or_wandb(
    offline_entrypoint, script, epochs
):
    trainer = offline_entrypoint.run(script)

    assert isinstance(
        trainer, IACTrainer if script == "tldr-len-ratio" else MAGRPOTrainer
    )
    assert trainer.args.num_train_epochs == epochs
    assert trainer.args.num_turns == 1
    assert len(trainer.agents) == 2
    assert trainer.wandb_config is None
    assert not offline_entrypoint.logged
    if script == "tldr-len-ratio":
        assert offline_entrypoint.saved == ["./iac_tldr"]
        assert trainer.formatters[0]({"prompt": "Example prompt"}).endswith(
            "\n\nExample prompt"
        )


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize(
    "options,expected",
    [
        (["--wandb"], {"project": None, "entity": "OpenMLRL", "name": None}),
        (
            [
                "--wandb-project",
                "custom-project",
                "--wandb-entity",
                "custom-workspace",
                "--wandb-run-name",
                "custom-run",
            ],
            {
                "project": "custom-project",
                "entity": "custom-workspace",
                "name": "custom-run",
            },
        ),
    ],
)
def test_explicit_wandb_options_reach_real_trainer(
    offline_entrypoint, script, options, expected
):
    trainer = offline_entrypoint.run(script, *options)

    assert trainer.wandb_initialized
    assert len(offline_entrypoint.logged) == 1
    assert {key: offline_entrypoint.logged[0][key] for key in expected} == expected


@pytest.mark.parametrize("script", SCRIPTS)
def test_example_model_and_epoch_overrides_reach_real_trainer(
    offline_entrypoint, script
):
    trainer = offline_entrypoint.run(
        script, "--model-name", "local-model", "--num-train-epochs", "1"
    )

    assert offline_entrypoint.loaded_models == ["local-model", "local-model"]
    assert trainer.args.num_train_epochs == 1


@pytest.mark.parametrize("script", SCRIPTS[1:])
def test_magrpo_examples_keep_config_validation(offline_entrypoint, script):
    with pytest.raises(ValueError, match="num_train_epochs must be >= 1"):
        offline_entrypoint.run(script, "--num-train-epochs", "0")


def test_tldr_keeps_existing_chat_template(offline_entrypoint):
    module = _load_example("tldr-len-ratio")
    tokenizer = offline_entrypoint.tokenizer
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] }}:{{ message['content'] }}|"
        "{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}"
    )

    prompt = module.build_prompt_formatters(tokenizer)[0]({"prompt": "Example prompt"})

    assert prompt.startswith("system:You summarize Reddit posts")
    assert prompt.endswith("|user:Example prompt|assistant:")
