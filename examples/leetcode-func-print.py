# Shuo: Tested 2 * Qwen2.5-1.5B model on a A100, at least 70GB VRAM is required

import ast
import contextlib
import io
import os
import re
import signal
from functools import partial
from typing import Any, Dict, List

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from comlrl.trainers.magrpo import MAGRPOConfig, MAGRPOTrainer


def cleanup_code(code):
    """Remove markdown code blocks and other non-executable parts from code."""
    code = re.sub(r"```python\s*", "", code)
    code = re.sub(r"```\s*", "", code)

    lines = code.split("\n")
    cleaned_lines = []
    for line in lines:
        if re.match(r"^[A-Z][^:=]*$", line.strip()) or re.match(
            r"^Here is", line.strip()
        ):
            continue
        if line.strip() and not re.match(
            r"^(import|from|def|class|if|else|elif|for|while|try|except|#|\s+|return|print)",
            line.strip(),
        ):
            continue
        cleaned_lines.append(line)

    code = "\n".join(cleaned_lines)

    code = re.split(
        r"In this example:|As you can see:|The function is defined|This function|We can use",
        code,
    )[0]

    code = re.sub(r"Example usage:", "# Example usage", code)
    code = re.sub(r"Now we can test", "# Test", code)

    return code.strip()


def extract_print_statements(code):
    """Extract print statements and their expected outputs from code."""
    # Find all print lines with expected output comments
    lines = code.split("\n")
    print_statements = []

    for line in lines:
        line = line.strip()
        if line.startswith("print(") and "#" in line:
            # Split by comment
            code_part, comment_part = line.split("#", 1)

            print_stmt = code_part.strip()

            expected_match = re.search(
                r"Expected\s+output:?\s*(.*)", comment_part, re.IGNORECASE
            )
            if expected_match:
                expected_output = expected_match.group(1).strip()
                print_statements.append((print_stmt, expected_output))
            elif comment_part.strip():
                expected_output = comment_part.strip()
                print_statements.append((print_stmt, expected_output))

    return print_statements


class TimeoutException(Exception):
    """Exception raised when code execution times out."""

    pass


def timeout_handler(signum, frame):
    """Signal handler for timeouts."""
    raise TimeoutException("Code execution timed out")


def deduplicate_tests(print_statements):
    """Deduplicate test cases by normalizing and comparing them."""
    unique_tests = []
    seen_tests = set()

    for stmt, expected in print_statements:
        func_call_match = re.search(r"print\((.*)\)", stmt)
        if func_call_match:
            func_call = func_call_match.group(1).strip()
            normalized_call = re.sub(r"\s+", "", func_call)

            if normalized_call not in seen_tests:
                seen_tests.add(normalized_call)
                unique_tests.append((stmt, expected))

    return unique_tests


def extract_imported_libraries(import_code):
    """Extract library names from import statements."""
    libraries = set()
    if not import_code:
        return libraries

    lines = import_code.split("\n")
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("import "):
            lib_part = line[7:].strip()
            for lib in lib_part.split(","):
                lib_name = lib.strip().split(".")[0].split(" as ")[0]
                if lib_name and not lib_name.startswith("#"):
                    libraries.add(lib_name)
        elif line.startswith("from "):
            match = re.match(r"from\s+(\w+)", line)
            if match:
                lib_name = match.group(1)
                libraries.add(lib_name)

    if libraries:
        print(f"DEBUG: Extracted libraries: {libraries}")
    else:
        print(f"DEBUG: No libraries extracted from code: '{import_code}'")

    return libraries


def check_library_usage(function_code, imported_libraries):
    """Check if the function code uses any of the imported libraries."""
    if not imported_libraries or not function_code:
        return False

    function_lower = function_code.lower()

    for lib in imported_libraries:
        if re.search(rf"\b{re.escape(lib.lower())}\b\.", function_lower):
            print(f"DEBUG: Found library usage: {lib} (dot notation)")
            return True

        common_aliases = {
            "numpy": ["np"],
            "pandas": ["pd"],
            "matplotlib": ["plt"],
            "seaborn": ["sns"],
        }

        if lib.lower() in common_aliases:
            for alias in common_aliases[lib.lower()]:
                if re.search(rf"\b{re.escape(alias)}\b\.", function_lower):
                    print(f"DEBUG: Found library usage: {lib} via alias {alias}")
                    return True

    print(f"DEBUG: No library usage found for {imported_libraries} in function code")
    return False


def execution_reward(completion1: List[str], completion2: List[str]) -> List[float]:
    """Reward: +0.5 for runnable code, +0.1 per correct test (max 5)."""
    rewards = []
    # Set timeout threshold (300 seconds for initial execution)
    TIMEOUT_SECONDS = 300

    for c1, c2 in zip(completion1, completion2):
        reward = 0.0

        c1_clean = cleanup_code(c1)
        c2_clean = cleanup_code(c2)

        print("\n--- Testing Code Generation ---")

        function_match = re.search(r"def\s+(\w+)\s*\(", c1_clean)
        if not function_match:
            print("No function definition found in first completion")
            rewards.append(reward)
            continue

        function_name = function_match.group(1)

        function_used = (
            re.search(r"\b" + re.escape(function_name) + r"\s*\(", c2_clean) is not None
        )
        function_redefined = (
            re.search(r"def\s+" + re.escape(function_name) + r"\s*\(", c2_clean)
            is not None
        )

        if not function_used:
            print(f"Function {function_name} is not used in second completion")
            rewards.append(reward)
            continue

        if function_redefined:
            print(f"Function {function_name} is redefined in second completion")
            rewards.append(reward)
            continue

        test_code = c2_clean if function_redefined else (c1_clean + "\n\n" + c2_clean)

        if len(test_code) < 1000:
            print("\n--- Combined Test Code ---")
            print(test_code)

        try:
            ast.parse(test_code)

            reward += 0.5
            print("✓ Syntax valid: +0.5 reward")

            print_statements = extract_print_statements(test_code)
            unique_print_statements = deduplicate_tests(print_statements)
            unique_print_statements = unique_print_statements[:5]

            if unique_print_statements:
                print(f"Found {len(unique_print_statements)} unique test(s) to run")

                local_vars = {}
                try:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(TIMEOUT_SECONDS)

                    exec(test_code, local_vars)

                    signal.alarm(0)
                    print(
                        f"✓ Initial code execution completed within {TIMEOUT_SECONDS}s"
                    )

                    for i, (print_stmt, expected_output) in enumerate(
                        unique_print_statements
                    ):
                        test_env = dict(local_vars)

                        try:
                            stdout_buffer = io.StringIO()
                            with contextlib.redirect_stdout(stdout_buffer):
                                exec(print_stmt, test_env)

                            actual_output = stdout_buffer.getvalue().strip()
                            expected_output = expected_output.strip()

                            def extract_number(text):
                                """Extract the first number from text, return as string"""
                                match = re.search(r"-?\d+(?:\.\d+)?", text)
                                return match.group(0) if match else text.strip()

                            expected_num = extract_number(expected_output)
                            actual_num = extract_number(actual_output)

                            outputs_match = expected_num == actual_num

                            result_symbol = "✓" if outputs_match else "✗"
                            print(f"Test {i + 1}: {result_symbol} Input: {print_stmt}")
                            print(
                                f"  Expected: {expected_output} | Actual: {actual_output}"
                            )

                            if outputs_match:
                                reward += 0.1
                                print(f"  Passed: +0.1 reward, current total: {reward}")

                        except Exception as e:
                            print(
                                f"Test {i + 1}: ✗ Input: {print_stmt} (Error: {str(e)})"
                            )

                except TimeoutException:
                    print(
                        f"✗ Initial code execution timed out after {TIMEOUT_SECONDS}s - failing all tests"
                    )
                    signal.alarm(0)

                except RecursionError as e:
                    print(
                        f"✗ Initial code execution failed due to recursion limit: {str(e)}"
                    )
                    signal.alarm(0)

                except Exception as e:
                    print(f"✗ Initial code execution failed: {str(e)}")
                    signal.alarm(0)
            else:
                print("No test prints with expected outputs found")

        except SyntaxError as e:
            print(f"✗ Syntax error: {str(e)}")
            rewards.append(reward)
            continue

        print(f"Final reward: {reward}")
        rewards.append(reward)

    return rewards


def create_code_dataset():
    """Create a small dataset with coding tasks."""
    data = {
        "question": [
            "Create a function that checks if a given string is a palindrome (reads the same forwards and backwards).",
            "Write a function that returns the nth Fibonacci number in the sequence (where F(0)=0, F(1)=1, and each subsequent number is the sum of the two preceding ones).",
            "Create a function that counts the number of vowels in a string.",
            "Write a function that checks if a number is prime.",
        ],
        "input": [
            "'radar', 'hello', 'lol'",
            "2, 5, 10",
            "'hello', 'world', 'aeiou'",
            "3, 6, 11",
        ],
        "output": [
            "True, False, True",
            "1, 5, 55",
            "2, 1, 5",
            "True, False, True",
        ],
        "function_name": [
            "is_palindrome",
            "fibonacci",
            "count_vowels",
            "is_prime",
        ],
    }
    return Dataset.from_dict(data)


def function_writer_formatter(example: Dict[str, Any]) -> str:
    """Format prompt for function-writing agent (Agent 1)."""
    question = example.get("question", "")
    function_name = example.get("function_name", "")
    input_example = example.get("input", "")
    output_example = example.get("output", "")

    prompt = (
        "You are asked to implement a Python function based on the following requirement:\n\n"
        f"Question: {question}\n\n"
        f"Function name: {function_name}\n\n"
    )

    if input_example:
        prompt += f"Example inputs: {input_example}\n\n"

    if output_example:
        prompt += f"Expected outputs: {output_example}\n\n"

    prompt += (
        "IMPORTANT REQUIREMENTS:\n\n"
        "1. Write a single, self-contained function with the exact name specified above.\n"
        "2. Include all helper code inside your main function - do not use unknown variables or functions.\n"
        "3. Ensure your function handles data types correctly - if numeric inputs are needed, convert string inputs to the right type.\n"
        "4. Include appropriate docstrings and comments.\n"
        "5. Make sure your function handles edge cases.\n"
        "6. Do not include any usage examples, main function, or explanations outside the function code.\n"
        "7. Do not add any explanatory text before or after the function - only provide the function code itself.\n\n"
        "Your solution must be completely self-contained within a single function."
    )

    return prompt


def example_writer_formatter(
    example: Dict[str, Any], generate_new_examples: bool = True
) -> str:
    """Format prompt for example-writing agent (Agent 2)."""
    question = example.get("question", "")
    function_name = example.get("function_name", "")
    input_example = example.get("input", "")
    output_example = example.get("output", "")

    input_type_hint = ""
    if input_example:
        inputs = input_example.split(",")
        if any(i.strip().isdigit() for i in inputs):
            input_type_hint = (
                "\nIMPORTANT: For numeric inputs, use integers without quotes"
            )

    prompt = (
        "You are asked to write usage examples for a Python function:\n\n"
        f"Question: {question}\n\n"
        f"Function name: {function_name}\n\n"
    )

    if input_example:
        prompt += f"Example inputs: {input_example}\n\n"

    if output_example:
        prompt += f"Expected outputs: {output_example}\n\n"

    prompt += (
        "IMPORTANT REQUIREMENTS:\n\n"
        "1. Write code that demonstrates how to use the function with ONLY real test examples.\n"
    )
    if generate_new_examples:
        prompt += "2. The test examples should include 3 cases above, and you need to design 2 edge cases by yourself, so 5 tests in total.\n"
    else:
        prompt += "2. The test examples should be the 3 cases above.\n"

    prompt += (
        "3. Assume the function has already been defined - DO NOT implement the function itself.\n"
        "4. Use actual values, not placeholders like 'example_input', 'input', 'result', etc.\n"
        "5. Include print statements with the function calls and expected output comments.\n"
        "6. Show a variety of inputs that test different cases.\n"
        "7. Expected output comments should contain only the result, no explanations.\n"
        f"{input_type_hint}\n\n"
        "Example format:\n"
        "```python\n"
        "# Example usage\n"
        f"print({function_name}(input))  # Expected output: result\n"
        "```\n\n"
        "REMEMBER: DO NOT redefine the function or include the function code - assume it already exists."
    )

    return prompt


def main():
    """Main function to run the experiment."""
    output_dir = "../../../projects/bepg/sliu30/output"
    os.makedirs(output_dir, exist_ok=True)

    train_dataset = create_code_dataset()

    model_name = "Qwen/Qwen2.5-1.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = MAGRPOConfig(
        output_dir=output_dir,
        num_train_epochs=100,
        per_device_train_batch_size=1,
        learning_rate=1e-5,
        logging_steps=1,
        save_steps=5,
        num_generations=4,
        max_new_tokens=256,
        temperature=0.8,
        top_p=0.9,
    )

    generate_new_examples = True
    configured_example_writer_formatter = partial(
        example_writer_formatter, generate_new_examples=generate_new_examples
    )

    wandb_config = {
        "project": "code-agents-magrpo",
        "entity": "OpenMLRL",
        "name": "3-basic-2-edge" if generate_new_examples else "3-basic",
    }

    trainer = MAGRPOTrainer(
        agents=[AutoModelForCausalLM.from_pretrained(model_name) for _ in range(2)],
        num_agents=2,
        reward_func=execution_reward,
        formatters=[function_writer_formatter, configured_example_writer_formatter],
        args=config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        wandb_config=wandb_config,
    )
    trainer.train()
    print("Training complete!")


if __name__ == "__main__":
    main()
