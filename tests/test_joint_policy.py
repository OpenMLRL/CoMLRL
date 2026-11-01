"""Joint-sequence log-probabilities and policy gradients for one joint actor."""

from types import SimpleNamespace

from datasets import Dataset
import pytest
import torch

from comlrl.trainers.pbrl import MARLHFConfig, MARLHFTrainer
from comlrl.trainers.centralized import (
    TaggedCentralizedComparatorAdapter,
    joint_sequence_log_prob,
)
from comlrl.trainers.reinforce import MAGRPOConfig, MAGRPOTrainer

from helpers import (
    BAD,
    GOOD,
    make_centralized_magrpo_trainer,
    make_preference_trainer,
    model_for,
)


@pytest.mark.parametrize("shared_eos", [False, True])
@pytest.mark.parametrize("length", [1, 6])
@pytest.mark.parametrize("dropout", [0.0, 0.3])
def test_joint_logprob_matches_all_tokens_and_gradients(
    tokenizer, shared_eos, length, dropout
):
    if shared_eos:
        tokenizer.pad_token = tokenizer.eos_token
    model = model_for(tokenizer)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = dropout
    prompt = torch.tensor(tokenizer.encode("joint left right"))
    target = torch.tensor(tokenizer.encode(GOOD)[:length] + [tokenizer.eos_token_id])
    with torch.inference_mode():
        padded = torch.cat([target, torch.full((3,), tokenizer.pad_token_id)])
    torch.manual_seed(23)
    logits = model(torch.cat([prompt, target[:-1]])[None], use_cache=False).logits[
        0, len(prompt) - 1 :
    ]
    expected = logits.float().log_softmax(-1).gather(1, target[:, None]).sum()
    expected_grads = torch.autograd.grad(expected, tuple(model.parameters()))
    torch.manual_seed(23)
    actual = joint_sequence_log_prob(model, tokenizer, prompt, padded)
    actual_grads = torch.autograd.grad(actual, tuple(model.parameters()))
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_empty_joint_response_allows_backward_with_frozen_embeddings(tokenizer):
    model = model_for(tokenizer)
    next(model.parameters()).requires_grad_(False)
    prompt = torch.tensor(tokenizer.encode("joint left right"))
    loss = joint_sequence_log_prob(
        model, tokenizer, prompt, torch.tensor([tokenizer.pad_token_id])
    )
    assert loss.item() == 0.0
    loss.backward()
    assert all(param.grad is None for param in model.parameters())
    with pytest.raises(ValueError, match="non-empty tokenized prompt"):
        joint_sequence_log_prob(model, tokenizer, prompt[:0], prompt[:1])


def test_marlhf_joint_loss_calls_sequence_log_prob_hook(
    tokenizer, tmp_path, monkeypatch
):
    trainer, _ = make_preference_trainer(
        MARLHFTrainer, MARLHFConfig, tokenizer, tmp_path
    )

    def require_mask(*args):
        raise ValueError("Custom completion mask is required")

    monkeypatch.setattr(trainer, "_sequence_log_prob", require_mask)
    data = {
        "prompt_input_ids": torch.tensor([[3]]),
        "completion_input_ids": [[torch.tensor([6])]],
    }
    with pytest.raises(ValueError, match="Custom completion mask"):
        trainer._compute_loss_with_gradients(trainer.agents[0], data, [1.0])


@pytest.mark.parametrize("learned_reward", [False, True])
@pytest.mark.parametrize("mode", ["mean", "raw", "max", "rloo"])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("reference_kl", [False, True])
def test_joint_policy_gradient_matches_manual_objective(
    tokenizer, tmp_path, learned_reward, mode, normalize, reference_kl
):
    options = dict(
        advantage_normalization=normalize,
        reference_kl_enabled=reference_kl,
        reference_devices=["cpu"],
        reference_kl_coef=0.3,
    )
    if learned_reward:
        trainer, _ = make_preference_trainer(
            MARLHFTrainer, MARLHFConfig, tokenizer, tmp_path, **options
        )
    else:
        trainer, _ = make_centralized_magrpo_trainer(tokenizer, **options)
    trainer.advantage_mode = mode
    model = trainer.agents[0]
    prompt = torch.tensor(tokenizer.encode("joint left right"))
    targets = [
        torch.tensor(tokenizer.encode(text) + [tokenizer.eos_token_id])
        for text in (GOOD, BAD)
    ]
    data = {
        "prompt_input_ids": prompt[None],
        "completion_input_ids": [
            [
                torch.cat([target, torch.full((idx + 1,), tokenizer.pad_token_id)])
                for idx, target in enumerate(targets)
            ]
        ],
        "reference_kls": [0.2, 0.7],
    }
    returns = torch.tensor([2.0, 0.5])
    if reference_kl:
        returns -= 0.3 * torch.tensor([0.2, 0.7])
    advantages = {
        "mean": returns - returns.mean(),
        "raw": returns,
        "max": returns - returns.max(),
        "rloo": 2 * returns - returns.sum(),
    }[mode]
    if normalize:
        advantages = (advantages - advantages.mean()) / advantages.std(
            unbiased=False
        ).clamp(min=1e-6)
    losses = []
    for target, advantage in zip(targets, advantages):
        logits = model(torch.cat([prompt, target[:-1]])[None], use_cache=False).logits[
            0, len(prompt) - 1 :
        ]
        losses.append(
            -logits.log_softmax(-1).gather(1, target[:, None]).sum() * advantage
        )
    expected = torch.stack(losses).mean()
    expected_grads = torch.autograd.grad(expected, tuple(model.parameters()))
    actual = trainer._compute_loss_with_gradients(model, data, [2.0, 0.5])
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual, tuple(model.parameters()))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("learned_reward", [False, True])
def test_joint_loss_validates_alignment_and_handles_empty_batch(
    tokenizer, tmp_path, learned_reward
):
    if learned_reward:
        trainer, _ = make_preference_trainer(
            MARLHFTrainer, MARLHFConfig, tokenizer, tmp_path
        )
    else:
        trainer, _ = make_centralized_magrpo_trainer(tokenizer)
    model = trainer.agents[0]
    next(model.parameters()).requires_grad_(False)
    data = {"prompt_input_ids": torch.tensor([[3]]), "completion_input_ids": [[]]}
    trainer._compute_loss_with_gradients(model, data, []).backward()
    with pytest.raises(ValueError, match="align with their returns"):
        trainer._compute_loss_with_gradients(model, data, [1.0])


@pytest.mark.parametrize("num_roles", [1, 3])
def test_joint_actor_trains_and_evaluates_with_variable_role_counts(
    tokenizer, num_roles
):
    tokenizer.add_tokens(["<agent_2>", "</agent_2>"])
    model = model_for(tokenizer)
    good = " ".join(f"<agent_{i}> good </agent_{i}>" for i in range(num_roles))
    bad = good.replace("good", "bad")

    class Adapter(TaggedCentralizedComparatorAdapter):
        def build_prompt(self, item, agent_prompts):
            assert len(agent_prompts) == num_roles
            return "joint left right"

    def generate(*, input_ids, num_return_sequences, **kwargs):
        return SimpleNamespace(
            sequences=torch.stack(
                [
                    torch.cat(
                        [
                            input_ids[0],
                            torch.tensor(tokenizer.encode(good if i % 2 == 0 else bad)),
                        ]
                    )
                    for i in range(num_return_sequences)
                ]
            )
        )

    model.generate = generate

    def reward(*roles, batch_items, **kwargs):
        assert len(roles) == num_roles
        assert batch_items == [{"prompt": "task"}]
        return [float(sum(role[0] == "good" for role in roles))]

    observed = []
    dataset = Dataset.from_list([{"prompt": "task"}])
    trainer = MAGRPOTrainer(
        agents=[model],
        num_agents=num_roles,
        tokenizer=tokenizer,
        args=MAGRPOConfig(
            collaboration_mode="centralized",
            num_turns=1,
            num_agents=num_roles,
            agent_devices="cpu",
            num_train_epochs=1,
            num_generations=2,
            eval_interval=0,
            rollout_buffer_size=1,
            train_batch_size=1,
        ),
        centralized_adapter=Adapter(),
        reward_func=reward,
        train_dataset=dataset,
        eval_dataset=dataset,
        eval_logger=lambda **kw: observed.append(kw) or [],
        eval_aggregator=lambda *a, **kw: {},
    )
    before = model.transformer.wte.weight.detach().clone()
    trainer.train()
    assert not torch.equal(before, model.transformer.wte.weight)
    assert trainer.num_agents == len(trainer.optimizers) == 1
    assert trainer.env_step == 2
    metrics = trainer.evaluate(num_eval_samples=1)
    assert metrics["eval/turn_1/reward_mean"] == num_roles
    assert observed[0]["agent_completions_turns"] == [[["good"]]] * num_roles


def test_generation_with_eos_in_prompt_preserves_joint_boundaries_and_kl(tokenizer):
    tokenizer.pad_token = tokenizer.eos_token
    trainer, _ = make_centralized_magrpo_trainer(
        tokenizer, reference_kl_enabled=True, reference_devices=["cpu"]
    )
    model = trainer.agents[0]
    trainer.formatters = [lambda _: "left <eos> right"]
    target = torch.tensor(tokenizer.encode(GOOD) + [tokenizer.eos_token_id])

    def generate(*, input_ids, num_return_sequences, **kwargs):
        full = torch.cat(
            [input_ids[0], target, torch.full((2,), tokenizer.pad_token_id)]
        )
        return SimpleNamespace(sequences=full[None].repeat(num_return_sequences, 1))

    model.generate = generate
    # Ensure the KL is nonzero, so an accidentally excluded EOS is observable.
    with torch.no_grad():
        model.transformer.wte.weight[tokenizer.eos_token_id].add_(0.5)
    data = trainer._generate_completions(
        model, [{"id": "task"}], num_return_sequences=2
    )
    prompt = data["prompt_input_ids"][0]
    assert prompt.tolist() == tokenizer.encode("left <eos> right")
    assert data["completions"] == [[GOOD, GOOD]]
    assert data["response_lens"] == [len(target), len(target)]
    torch.testing.assert_close(
        data["completion_input_ids"][0][0][: len(target)], target
    )
    full = torch.cat([prompt, target])[None]
    model.eval()
    policy_logits = model(full).logits[0, len(prompt) - 1 : -1]
    reference_logits = trainer.reference_models[0](full).logits[0, len(prompt) - 1 : -1]
    log_ratio = (
        reference_logits.log_softmax(-1) - policy_logits.log_softmax(-1)
    ).gather(1, target[:, None])
    expected = (log_ratio.exp() - log_ratio - 1).sum().item()
    assert data["reference_kls"] == pytest.approx([expected, expected], abs=1e-6)


def test_real_generation_can_feed_joint_backward(tokenizer):
    torch.manual_seed(19)
    trainer, _ = make_centralized_magrpo_trainer(tokenizer)
    model = trainer.agents[0]
    del model.generate
    trainer.advantage_mode = "raw"
    trainer.args.advantage_normalization = False
    data = trainer._generate_completions(
        model,
        [{"id": "task"}],
        num_return_sequences=2,
        max_new_tokens=4,
        min_new_tokens=4,
    )
    loss = trainer._compute_loss_with_gradients(model, data, [1.0, 2.0])
    assert torch.isfinite(loss)
    loss.backward()
    assert model.transformer.wte.weight.grad.abs().sum() > 0
