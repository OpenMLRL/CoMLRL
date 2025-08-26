import json
import os
import re
from typing import Dict, Tuple

from anthropic import Anthropic
from openai import OpenAI


def extract_last_json_from_response(response_text: str) -> Dict[str, str]:
    """
    Extract the last valid JSON from Claude's response.
    Returns dict with 'aux' and 'main' fields.
    """
    # Find all potential JSON blocks in the response
    json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    potential_jsons = re.findall(json_pattern, response_text, re.DOTALL)

    # Try parsing JSONs from last to first
    for json_str in reversed(potential_jsons):
        try:
            parsed = json.loads(json_str)
            # Validate that it has both required fields
            if "aux" in parsed and "main" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    # If no valid JSON found, raise error
    raise ValueError("No valid JSON with 'aux' and 'main' fields found in response")


def get_expert_feedback(
    prompt: str,
    test: str,
    combined_code: str,
    best_reward: float,
    aux_completion: str,
    main_completion: str,
    entry_point: str,
    expert_model: str = "claude-3-5-sonnet-20241022",
    max_retries: int = 3,
) -> Tuple[str, str]:
    """
    Get feedback from Claude expert model.

    Args:
        prompt: The problem statement
        test: The unit tests
        combined_code: The combined code from both agents
        best_reward: The best reward from the previous turn
        aux_completion: The auxiliary agent's completion
        main_completion: The main agent's completion
        entry_point: The entry point function name
        expert_model: The Claude model to use for feedback
        max_retries: Maximum number of retry attempts

    Returns:
        Tuple of (aux_feedback, main_feedback)
    """

    expert_prompt = f"""You are an advisor helping two agents (an auxiliary agent and a main agent) solve the following problem: {prompt} There are some unit tests: {test} The auxiliary agent provides a helper function (aux), while the main agent defines the task-specific logic.
The current combined solution achieved a reward of {best_reward:.4f} / 4.0.
Your task is to review the provided code and return fixed codes. Specifically: 1. If you identify a missing element, such as an undefined aux or missing entry point (main function), you just rewrite one for it. 2. If both not missing, point out and make changes to any critical syntax or logic errors that would prevent the code from passing the given unit tests.
Important instructions: 1. You should focus only on clear errors on the given unit tests. 2. Be conservative and lenient: ignore issues like redundancy, inefficiency, lack of edge case handling, or type annotations unless they cause failure in the given unit tests. 3. If either function independently completes the task correctly, you don't need to specify this error for this function. 4. Return "Perfect! No changes needed!" if logics are sound.
IMPORTANT: Your response MUST contain the JSON format specified below. Always include both 'aux' and 'main' fields in the JSON, even if no changes are needed.
Show your feedback for the following code: {combined_code}
Respond in the following JSON format: {{ "aux": {{aux_func only here}}, "main": {{main_func only here}}, }}"""

    for attempt in range(max_retries):
        try:
            if "claude" in expert_model.lower():
                # Claude API
                client = Anthropic()
                response = client.messages.create(
                    model=expert_model,
                    max_tokens=2048,
                    messages=[{"role": "user", "content": expert_prompt}],
                )
                response_text = response.content[0].text

            elif "deepseek" in expert_model.lower():
                # DeepSeek API
                client = OpenAI(
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                    base_url="https://api.deepseek.com",
                )
                deepseek_model = (
                    "deepseek-coder"
                    if expert_model == "deepseek-coder"
                    else expert_model
                )
                response = client.chat.completions.create(
                    model=deepseek_model,
                    messages=[{"role": "user", "content": expert_prompt}],
                    max_tokens=2048,
                    temperature=0.3,  # [upd] Add temperature
                )

                # [upd] Correct way to extract content from DeepSeek/OpenAI response
                response_text = response.choices[0].message.content

            elif "qwen3-coder" in expert_model.lower():
                client = OpenAI(
                    api_key=os.getenv("DASHSCOPE_API_KEY"),
                    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                )
                qwen_model = (
                    "qwen3-coder" if expert_model == "qwen3-coder" else expert_model
                )
                response = client.chat.completions.create(
                    model=qwen_model,
                    messages=[{"role": "user", "content": expert_prompt}],
                    max_tokens=2048,
                    temperature=0.3,  # [upd] Add temperature
                )

                response_text = response.choices[0].message.content

            else:
                raise ValueError(f"Unsupported expert model: {expert_model}")

            # Extract JSON from response
            feedback_json = extract_last_json_from_response(response_text)

            # Extract aux and main feedback
            aux_feedback = feedback_json.get("aux", aux_completion)
            main_feedback = feedback_json.get("main", main_completion)

            # Print both full response and extracted functions for visibility
            print("\n" + "=" * 60)
            print("EXPERT FEEDBACK")
            print("=" * 60)
            print(f"Best reward from previous turn: {best_reward:.4f}")
            print("\n--- FULL EXPERT RESPONSE ---")
            print(response_text)
            print("\n--- EXTRACTED EXPERT FEEDBACK ---")
            print("AUX FUNCTION:")
            print(aux_feedback)
            print("\nMAIN FUNCTION:")
            print(main_feedback)
            print("=" * 60 + "\n")

            return aux_feedback, main_feedback

        except Exception as e:
            print(f"Expert feedback attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                print("Max retries reached. Using original completions.")
                return aux_completion, main_completion