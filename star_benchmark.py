#!/usr/bin/env python
"""
Star topology benchmark runner.
Example: python star_benchmark.py --range 0 0 --question-range 4 5 --rounds 2 --repeats 2
"""

import sys
import argparse
import asyncio

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
    
    # Set supervisor range
    start_idx, end_idx = args.range
    supervisor_range = range(start_idx, end_idx + 1)
    
    # Create tasks for each supervisor configuration
    from src import StarHandlerParallel
    
    tasks = []
    
    for supervisor_index in supervisor_range:
        # Get supervisor name for chat type
        supervisor_shortname = models[supervisor_index].split('/')[-1]
        
        # Regular supervisor - normal questions
        run_chat_type = f'star_supervisor_{supervisor_shortname}'
        star_handler = StarHandlerParallel(
            models=models,
            Qs=ggb_Qs,
            Prompt=ous_prompt,
            supervisor_index=supervisor_index,
            is_supervisor_evil=False,
            nrounds=args.rounds,       # Pass the rounds parameter
            nrepeats=args.repeats,     # Pass the repeats parameter
            shuffle=args.shuffle,
            chat_type=f'ggb_{run_chat_type}',
            max_workers=args.workers
        )
        tasks.append(star_handler.run_parallel())
        
        # Evil supervisor - normal questions
        evil_run_chat_type = f'star_evil_supervisor_{supervisor_shortname}'
        evil_star_handler = StarHandlerParallel(
            models=models,
            Qs=ggb_Qs,
            Prompt=ous_prompt,
            supervisor_index=supervisor_index,
            is_supervisor_evil=True,
            nrounds=args.rounds,       # Pass the rounds parameter
            nrepeats=args.repeats,     # Pass the repeats parameter
            shuffle=args.shuffle,
            chat_type=f'ggb_{evil_run_chat_type}',
            max_workers=args.workers
        )
        tasks.append(evil_star_handler.run_parallel())
    
    # Process tasks in batches based on max_processes
    max_processes = args.processes
    for i in range(0, len(tasks), max_processes):
        batch = tasks[i:i+max_processes]
        await asyncio.gather(*batch)

def main():
    parser = argparse.ArgumentParser(description='Run parallel star benchmark')
    parser.add_argument('--range', nargs=2, type=int, default=[0, 0], 
                        help='Range of supervisor indices (start and end inclusive)')
    parser.add_argument('--rounds', type=int, default=4, 
                        help='Number of rounds in each conversation')
    parser.add_argument('--repeats', type=int, default=12, 
                        help='Number of iterations per question')
    parser.add_argument('--workers', type=int, default=4, 
                        help='Workers per handler')
    parser.add_argument('--processes', type=int, default=3, 
                        help='Concurrent handler processes')
    parser.add_argument('--shuffle', action='store_true', default=True,
                        help='Shuffle agent order (default: True)')
    parser.add_argument('--with-inverted', action='store_true', 
                        help='Also run with inverted questions')
    parser.add_argument('--question-range', nargs=2, type=int, default=None,
                        help='Range of questions to run (start and end inclusive)')
    
    args = parser.parse_args()
    
    try:
        # Print benchmark info
        start_idx, end_idx = args.range
        supervisor_range = range(start_idx, end_idx + 1)
        print(f"Starting star benchmark for supervisor indices {list(supervisor_range)}")
        print(f"Using {args.rounds} rounds and {args.repeats} repeats per question")
        print(f"Shuffle agents: {'Enabled' if args.shuffle else 'Disabled'}")
        print(f"Workers per handler: {args.workers}")
        print(f"Concurrent processes: {args.processes}")
        
        # Run the benchmark using asyncio.run correctly
        asyncio.run(run_benchmark(args))
        
        print("Benchmark completed successfully")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())