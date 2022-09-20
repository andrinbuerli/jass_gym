# Schieber Jass Gym
A gym wrapping the jass-kit-cpp Schieber Jass implementation.
Enables easy training leveraging [rllib](https://docs.ray.io/en/latest/rllib/index.html).

## Setup
Install the package

```bash
$ pip install -v -e .
```

Run tests to verify local setup

```bash
$ sjmz (--nodocker) --test
```

## Training
Train a Schieber Jass agent using rllib with

```bash
$ sjgym --file /app/resources/apex_multiagent.yaml --log
```

## Export
To export agent to [JIT](https://pytorch.org/docs/stable/jit.html) format 

```bash
$ sjgym --file /app/resources/apex_multiagent.yaml --export --latest_checkpoint <path-to-checkpoints>
```