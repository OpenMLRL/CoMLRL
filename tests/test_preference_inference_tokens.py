import pytest
import torch

from comlrl.trainers.preference import MADPOConfig, MADPOTrainer
from test_centralized_collaboration import model_for, tokenizer


@pytest.mark.parametrize("source", ["inference", "generation"])
def test_decentralized_logprob_accepts_inference_tokens_without_changing_objective(
    tokenizer, source
):
    torch.manual_seed(19)
    model = model_for(tokenizer)
    trainer = MADPOTrainer(
        agents=[model],
        num_agents=1,
        tokenizer=tokenizer,
        args=MADPOConfig(num_agents=1, agent_devices="cpu"),
        formatters=lambda _: "joint left right",
        reward_func=lambda completions: [0.0],
    )
    if source == "generation":
        data = trainer._generate_completions(
            model,
            [{}],
            num_return_sequences=1,
            max_new_tokens=4,
            min_new_tokens=4,
        )
        prompt = data["prompt_input_ids"][0]
        completion = data["completion_input_ids"][0][0]
    else:
        with torch.inference_mode():
            prompt = torch.tensor(tokenizer.encode("joint left right"))
            completion = torch.tensor(tokenizer.encode("good bad good bad"))
    assert completion.is_inference()

    normal_prompt = prompt.clone()
    normal_completion = completion.clone()
    expected = trainer._sequence_log_prob(0, normal_prompt, normal_completion)
    expected_gradients = torch.autograd.grad(expected, tuple(model.parameters()))
    actual = trainer._sequence_log_prob(0, prompt, completion)
    actual_gradients = torch.autograd.grad(actual, tuple(model.parameters()))

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
    torch.testing.assert_close(prompt, normal_prompt, rtol=0, atol=0)
    torch.testing.assert_close(completion, normal_completion, rtol=0, atol=0)
