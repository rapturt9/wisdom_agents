# Wisdom Agents

## Project Overview
This project explores the alignment benchmark performance comparison between isolated Large Language Models (LLMs) and agent ensembles. We investigate whether a collective of AI agents can achieve better performance on alignment benchmarks compared to individual models working in isolation.

## Research Questions
- Does polling different model types achieve better ensemble performance than any single model alone?
- How do different agent topologies (star vs. ring/chain) affect collective decision-making?
- Do stronger/newer models dominate the conversation in multi-agent systems?
- Can agents change their responses based on reasoning from other agents?
- How does the introduction of misaligned agents impact overall alignment?

## Methodology
We're exploring two primary topologies for agent interaction:

1. **Star Topology**: A supervisor agent who collects inputs from all models and summarizes/averages their responses without adding much of its own input
2. **Ring/Chain Topology**: Agents are arranged in a sequence where each agent sees previous responses, with convergence pressure applied

Our approach includes:
- Establishing baselines for all models in isolation
- Comparing performance across different model types and families
- Testing different agent configurations and interaction patterns
- Measuring confidence and variance in responses

## Implementation

### Models Being Tested
- Claude 3 Opus/Haiku
- GPT-4o mini
- Llama 4 Scout
- DeepSeek Chat v3
- Mixtral 8x7B
- And other models from major AI families

### Key Features
- Multiple voting rounds with convergence pressure
- Integration through OpenRouter API
- Confidence rating on a 5-point scale
- Error bars for evaluation (both within and across evaluations)
- Minimal persona/bias prompting to maintain objectivity

## Setup and Usage
```python
# Implementation details will be added here
```

## Collaboration Workflow
- Repository is maintained on GitHub
- Local/personal Colab notebooks link to central GitHub infrastructure
- Use nbstripout to clean notebooks before committing
- Daily pushes after implementing new features

## Development Setup
1. Install nbstripout and pre-commit:
   pip install nbstripout pre-commit
2. Set up the pre-commit hooks:
   pre-commit install

## Research Findings
Our preliminary investigations show:
- Agents can be convinced to change their answers based on reasoning from other agents
- Temperature settings impact model variance
- Prompting significantly affects question answers
- Later agents in chains can see and act on prior votes without "prompt poisoning"

## Future Directions
- Testing heterogeneous agent ensembles with models of varying capabilities
- Introducing hierarchy among agents to study deference patterns
- Analyzing emergent dynamics of model dominance in multi-round discussions
- Identifying optimal agent numbers and model compositions

## License
MIT
