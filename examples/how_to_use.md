# How to Use CoMLRL

CoMLRL is designed for LLM collaboration with multiple MARL algorithms. To set up a training script, you need to have:

## Reward Model

A reward model is a single callable that takes all agents' completions and returns a list of scalar rewards (one per sample). If you want to combine multiple criteria, wrap them inside your own composite function. You can optionally pass a single reward processor (callable) to post-process the scalar reward (e.g., scaling or shifting).

```
todo
```

## Dataset

you can design your own dataset like ... or a simple way is to use huggingface portal by just giving a string

### Configuration and Trainer

what are the necessary configurations?

what are given to trainer?
