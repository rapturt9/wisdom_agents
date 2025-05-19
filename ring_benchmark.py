#!/usr/bin/env python
"""
Ring topology benchmark runner.
Example: python ring_benchmark.py --range 0 0 --question-range 4 5 --rounds 2 --repeats 2
"""

import sys
import argparse
import asyncio
import os
import time
import json
from datetime import datetime

# Import what we need from src
from src import models, GGB_Statements, PromptHandler

# Hardcoded paths - change these if needed
QUESTION_JSON = 'GGB_benchmark/GreatestGoodBenchmark.json'
INVERTED_JSON = 'GGB_benchmark/GreatestGoodBenchmarkInverted.json'

async def run_benchmark(args):
    """Async function to run the benchmark"""
    # Load questions
    ggb_Qs = GGB_Statements(QUESTION_JSON)
    
    # Override question range 
    if args.question_range:
        q_start, q_end = args.question_range
        print(f"Using question range: {q_start}-{q_end}")
        # Directly set the question range - only process these questions
        ggb_Qs.QUESTION_RANGE = (q_start, q_end)
    
    # Create prompts
    ous_prompt = PromptHandler(group_chat=True)
    
    # Set model range
    start_idx, end_idx = args.range
    my_range = range(start_idx, end_idx + 1)
    
    # Create tasks for each model configuration
    from src import RingHandlerParallel
    
    n_models = len(models)
    tasks = []
    
    for i in my_range:
        if i == 0:
            run_models = models
            run_chat_type = 'hetero_ring'
        else:
            run_models = [models[i-1]] * n_models
            model_shortname = models[i-1].split('/')[-1]
            run_chat_type = f'{model_shortname}_ring'
        
        # Create the handler with the specific parameters
        ring = RingHandlerParallel(
            models=run_models,
            Qs=ggb_Qs,
            Prompt=ous_prompt,
            nrounds=args.rounds,     # Pass the rounds parameter
            nrepeats=args.repeats,   # Pass the repeats parameter
            shuffle=args.shuffle,
            chat_type=f'ggb_{run_chat_type}',
            max_workers=args.workers
        )
        
        # Store the task
        tasks.append(ring.run_parallel())
    
    # Run all tasks concurrently
    await asyncio.gather(*tasks)

def main():
    parser = argparse.ArgumentParser(description='Run parallel ring benchmark')
    parser.add_argument('--range', nargs=2, type=int, default=[0, 0], 
                        help='Range of model indices (start and end inclusive)')
    parser.add_argument('--rounds', type=int, default=4, 
                        help='Number of rounds in each conversation')
    parser.add_argument('--repeats', type=int, default=12, 
                        help='Number of iterations per question')
    parser.add_argument('--shuffle', action='store_true', default=True,
                        help='Shuffle agent order (default: True)')
    parser.add_argument('--with-inverted', action='store_true', 
                        help='Also run with inverted questions')
    parser.add_argument('--question-range', nargs=2, type=int, default=None,
                        help='Range of questions to run (start and end inclusive)')
    parser.add_argument('--workers', type=int, default=8,
                       help='Maximum number of workers per handler')
    
    args = parser.parse_args()
    
    try:
        # Print benchmark info
        start_idx, end_idx = args.range
        my_range = range(start_idx, end_idx + 1)
        print(f"Starting ring benchmark for model indices {list(my_range)}")
        print(f"Using {args.rounds} rounds and {args.repeats} repeats per question")
        print(f"Shuffle agents: {'Enabled' if args.shuffle else 'Disabled'}")
        print(f"Workers per handler: {args.workers}")
        
        # Run the benchmark using asyncio.run correctly
        asyncio.run(run_benchmark(args))
        
        print("Benchmark completed successfully")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())