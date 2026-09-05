# Centralized MAGRPO

`CentralizedMAGRPOConfig` and `CentralizedMAGRPOTrainer` are available from
`comlrl.trainers.reinforce`. They optimize a single joint policy directly with a
task reward, without preference generation, a comparator, or reward-model fitting.
The ordinary `MAGRPOConfig` and `MAGRPOTrainer` retain their decentralized behavior.

Pass `centralized_adapter` with `build_prompt(item, agent_prompts)` and
`parse_completion(completion, item, num_agents)` methods. Downstream applications
own the domain-specific prompts and parsers; the existing centralized comparator
adapters implement this interface and can be reused.

`num_agents` remains the logical role count. Supply one model source (or a
single-element `agents` list), one tokenizer, and one actor device. The trainer
uses one optimizer and scores every joint-response token, including the last.
Only the task reward and evaluation logger receive the parsed role responses.

Only `num_turns=1` is supported. `max_new_tokens` is the total joint-output budget;
`num_generations` counts complete joint responses. Each sampled joint response
adds one environment step. Single-turn steps are therefore
`dataset_size * num_generations * num_train_epochs` for a complete run.
