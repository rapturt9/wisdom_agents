# Wisdom Agents

## Project Overleaf Document (View Only):
https://www.overleaf.com/read/fpjbnfzymdbg#b92579

## Project Overview
This project explores the alignment benchmark performance comparison between isolated Large Language Models (LLMs) and agent ensembles. We investigate whether a collective of AI agents can achieve better performance on alignment benchmarks compared to individual models working in isolation.
Our focus is on creating test cases that resemble use cases of agentic ensembles in industry applications, e.g. update code bases, determine healthcare treatment, or execute research. In all use cases it is expected that agent ensembles will at some point face decisions to make without availability of direct user guidance, and our goal is to illuminate safety and alignment-related properties of such decision-making by example of a common alignment benchmark.  

## Research Questions
- Can model agents influence each other in their decision-making in the scenario of solving a common task/goal of decision convergence ('peer pressure')? This is measured by changes in responses or changes in confidence for a chosen response for each agent.
- Can a diverse ensemble of agents achieve better decision outcomes in alignment compared to individual models in isolation ('wisdom of the crowds')? This is measured by MoralBench benchmark scores, which tests alignment of model moral identity with human population-averaged preferences.
- Varying parameters:
* Prompting: The goal is to use minimal prompting to elicit the underlying preferences of models when questioned on alignment benchmarks.
* Decision-making topologies: How do different agent topologies (star vs. ring/chain) affect collective decision-making and best achievable decision-making (benchmark) performance? Do hierarchies introduced by agent responses (e.g. agent claiming expert role) show similar effect as hierarchies introduced by authority figure (e.g. developer, user, or supervisor agent)? Do models lower in the hierarchy show sycophantic tendencies towards models higher in the hierarchy?
* Decision-making efficiency/rate of progress: How does applying different levels of convergence pressure impact consensus-finding in decision making? 
* Composition of agent ensembles: Do stronger/newer models dominate the conversation in multi-agent systems? Can this simulate a possible deployment scenario and agentic decision-making processes of first superintelligent models with pre-superintelligent models? Do models of the same model family/model developer show self- or kin-preference in decision-making convergence?
* Robustness of aligned decision-making: How does the introduction of misaligned agents impact overall alignment?
* Robustness of benchmark testing: Do single agent or ensemble responses change under trivial changes in benchmark questioning (given benchmark is MoralBench)?

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

## Research Findings (results will be collected in offline research reports/Overleaf project)
Our preliminary investigations show:
- Agents can be prompted to change their answers based on reasoning from other agents
- Temperature settings does not impact model variance
- Prompting significantly affects question answers
- Later agents in chains can see and act on prior agent responses, but do not see prior agent prompting (no "prompt poisoning" from prior agents to later agents)

## Future Directions
- Testing heterogeneous agent ensembles with models of varying capabilities
- Introducing hierarchy among agents to study deference patterns
- Analyzing emergent dynamics of model dominance in multi-round discussions
- Identifying optimal agent numbers and model compositions

## License
MIT
