# Wisdom Agents

Arxiv Link: [MAEBE: Multi-Agent Emergent Behavior Framework
](https://arxiv.org/abs/2506.03053)

## Project Overview

This project explores the alignment benchmark performance comparison between isolated Large Language Models (LLMs) and agent ensembles. We investigate whether a collective of AI agents can achieve better performance on alignment benchmarks compared to individual models working in isolation.
Our focus is on creating test cases that resemble use cases of agentic ensembles in industry applications, e.g. update code bases, determine healthcare treatment, or execute research. In all use cases it is expected that agent ensembles will at some point face decisions to make without availability of direct user guidance, and our goal is to illuminate safety and alignment-related properties of such decision-making by example of a common alignment benchmark.

## Research Questions

- Can model agents influence each other in their decision-making in the scenario of solving a common task/goal of decision convergence ('peer pressure')? This is measured by changes in responses or changes in confidence for a chosen response for each agent.
- Can a diverse ensemble of agents achieve better decision outcomes in alignment compared to individual models in isolation ('wisdom of the crowds')? This is measured by GGB benchmark scores, which tests alignment of model moral identity with human population-averaged preferences.

## Methodology

We're exploring two primary topologies for agent interaction:

1. **Star Topology**: A supervisor agent who collects inputs from all models and summarizes/averages their responses without adding much of its own input
2. **Ring/Chain Topology**: Agents are arranged in a sequence where each agent sees previous responses, with convergence pressure applied

Our approach includes:

- Establishing baselines for all models of a diverse model set from different developers, in isolation
- Comparing performance across different model types and families
- Testing different agent configurations and interaction patterns
- Measuring confidence and variance in responses

## Implementation

### Models Being Tested from Openrouter

- `openai/gpt-4o-mini`
- `anthropic/claude-3.5-haiku`
- `google/gemini-2.0-flash-lite-001`
- `qwen/qwen-2.5-7b-instruct`
- `meta-llama/llama-3.1-8b-instruct`
- `deepseek/deepseek-chat-v3-0324`

### Key Features

- Multiple voting rounds with convergence pressure
- Integration through OpenRouter API
- Likert score output by LLM
- Error bars for evaluation (both within and across evaluations)
- Minimal persona/bias prompting to maintain objectivity

## Setup and Usage

### Python Scripts

- `analysis_functions.py`: Contains functions for analyzing benchmark results, including statistical analysis and aggregation of agent responses.
- `BufferedContext.py`: Implements buffered context management for agent interactions, useful for handling multi-turn conversations and context windows.
- `grade.py`: Grades agent responses against benchmarks. **Command-line arguments:**
  - `--input <file>`: Path to the input results file (JSON/CSV).
  - `--output <file>`: Path to save grading results.
  - `--benchmark <name>`: Specify which benchmark to use for grading.
- `helpers.py`: Utility functions for data processing, formatting, and API integration.
- `parallel_handlers.py`: Manages parallel execution of agent tasks and benchmarks, optimizing for speed and resource usage.
- `ring_benchmark.py`: Runs the ring/chain topology benchmark. **Command-line arguments:**
  - `--models <list>`: Comma-separated list of model names to use.
  - `--rounds <int>`: Number of voting rounds.
  - `--output <file>`: Path to save results.
- `src.py`: Main entry point for running agent ensemble experiments. **Command-line arguments:**
  - `--topology <star|ring>`: Selects the agent interaction topology.
  - `--models <list>`: Comma-separated list of models.
  - `--benchmark <file>`: Path to benchmark file.
  - `--output <file>`: Path to save results.
- `star_benchmark.py`: Runs the star topology benchmark. **Command-line arguments:**
  - `--models <list>`: Comma-separated list of model names.
  - `--output <file>`: Path to save results.
- `evil_only_star_benchmark.py`: Specialized star topology benchmark focusing on 'evil-only' scenarios. **Command-line arguments:**
  - `--models <list>`: Comma-separated list of models.
  - `--output <file>`: Path to save results.
- `run_single_agents.ipynb`: Jupyter notebook for running and analyzing individual agent responses on benchmarks. Includes code for loading models, running benchmarks, and visualizing results.

### Jupyter Notebooks

- `main.ipynb`: Central notebook for orchestrating experiments, running benchmarks, and aggregating results. Includes code for running both star and ring topologies, visualizing performance, and comparing agent ensembles.
- `multiagent.ipynb`: Explores multi-agent interaction patterns, convergence analysis, and peer pressure effects. Contains code for running experiments and plotting results.
- `poster_figures.ipynb`: Generates publication-quality figures and visualizations for research posters and papers, using results from benchmarks.
- `graph_graded_results.ipynb`: Visualizes graded results, including error bars, confidence intervals, and comparative plots across models and topologies.

```

```
