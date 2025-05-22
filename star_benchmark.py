#!/usr/bin/env python
"""
Star topology benchmark runner with improved error handling and range enforcement.
Example: python star_benchmark.py --range 0 0 --rounds 4 --repeats 12
"""

import sys
import argparse
import asyncio
import os
import time
import random
from datetime import datetime

# Import what we need from src
from src import models, GGB_Statements, PromptHandler
from parallel_handlers import StarHandlerParallel

# Hardcoded paths - change these if needed
QUESTION_JSON = 'GGB_benchmark/GreatestGoodBenchmark.json'
INVERTED_JSON = 'GGB_benchmark/GreatestGoodBenchmarkInverted.json'

async def run_benchmark(args):
    """Async function to run the benchmark"""
    # Set supervisor range
    start_idx, end_idx = args.range
    supervisor_range = range(start_idx, end_idx + 1)
    
    tasks = []
    
    # Determine which question sets to run
    if args.only_inverted:
        question_sets = [
            {'json_file': INVERTED_JSON, 'inverted': True, 'suffix': '_inverted'},
            ]
    else: 
        question_sets = [
            {'json_file': QUESTION_JSON, 'inverted': False, 'suffix': ''},
        ]
        
        if args.with_inverted:
            question_sets.append(
                {'json_file': INVERTED_JSON, 'inverted': True, 'suffix': '_inverted'}
            )
        
    # Process each question set
    for question_set in question_sets:
        # Load questions
        ggb_Qs = GGB_Statements(question_set['json_file'])
        ggb_prompt = PromptHandler(group_chat=True, invert_answer=question_set['inverted'])
        
        # Get question range boundaries
        if args.question_range:
            q_start, q_end = args.question_range
            print(f"Using question range: {q_start}-{q_end} for {os.path.basename(question_set['json_file'])}")
            # Set directly on the question object for consistency with original code
            ggb_Qs.QUESTION_RANGE = (q_start, q_end)
        else:
            print(f"Using all questions from {os.path.basename(question_set['json_file'])}")
            q_start, q_end = 1, ggb_Qs.get_total_questions()
        
        # Create tasks for each supervisor configuration
        for supervisor_index in supervisor_range:
            # Get supervisor name for chat type
            supervisor_shortname = models[supervisor_index].split('/')[-1]
            
            # Regular supervisor - normal questions
            run_chat_type = f'star_supervisor_{supervisor_shortname}'
            star_handler = StarHandlerParallel(
                models=models,
                Qs=ggb_Qs,
                Prompt=ggb_prompt,
                supervisor_index=supervisor_index,
                is_supervisor_evil=True,
                save_central_messages=True,
                nrounds=args.rounds,       
                nrepeats=args.repeats,     
                shuffle=args.shuffle,
                random_seed=None,  # No need for fixed seed anymore
                chat_type=f'ggb_{run_chat_type}{question_set["suffix"]}',
                max_workers=args.workers,
                max_concurrent_clients=args.concurrent_clients,
                rate_limit_window=args.rate_window,
                max_requests_per_window=args.rate_limit
            )
            # Explicitly set question range on the handler
            star_handler.QUESTION_RANGE = (q_start, q_end)
            tasks.append(star_handler.run_parallel())
    
    # Run all tasks concurrently or in batches if requested
    if args.batch_mode and len(tasks) > 1:
        # Process tasks in batches
        batch_size = min(args.batch_size, len(tasks))
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1} ({len(batch)} handlers)")
            await asyncio.gather(*batch)
    else:
        # Run all tasks concurrently
        await asyncio.gather(*tasks)

def main():
    parser = argparse.ArgumentParser(description='Run parallel star benchmark')
    parser.add_argument('--range', nargs=2, type=int, default=[0, 0], 
                        help='Range of supervisor indices (start and end inclusive)')
    parser.add_argument('--rounds', type=int, default=4, 
                        help='Number of rounds in each conversation')
    parser.add_argument('--repeats', type=int, default=12, 
                        help='Number of iterations per question')
    parser.add_argument('--workers', type=int, default=1,
                        help='Maximum number of workers per handler')
    parser.add_argument('--shuffle', action='store_true', default=True,
                        help='Shuffle agent order (default: True)')
    parser.add_argument('--with-inverted', action='store_true', 
                        help='Run with both standard AND inverted questions')
    parser.add_argument('--question-range', nargs=2, type=int, default=None,
                        help='Range of questions to run (start and end inclusive)')
    parser.add_argument('--concurrent-clients', type=int, default=10,
                        help='Maximum number of concurrent API clients')
    parser.add_argument('--rate-window', type=int, default=60,
                        help='Rate limit window in seconds')
    parser.add_argument('--rate-limit', type=int, default=100,
                        help='Maximum requests per rate limit window')
    parser.add_argument('--batch-mode', action='store_true',
                        help='Process handlers in batches rather than all concurrently')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Number of handlers to run concurrently in batch mode')
    parser.add_argument('--only-inverted', action='store_true', 
                        help='Run with only inverted questions')
    
    args = parser.parse_args()
    
    try:
        # Print benchmark info
        start_idx, end_idx = args.range
        supervisor_range = range(start_idx, end_idx + 1)
        print(f"Starting star benchmark for supervisor indices {list(supervisor_range)}")
        print(f"Using {args.rounds} rounds and {args.repeats} repeats per question")
        print(f"Question sets: {'Standard + Inverted' if args.with_inverted else 'Standard only'}")
        print(f"Shuffle agents: {'Enabled' if args.shuffle else 'Disabled'}")
        print(f"Workers per handler: {args.workers}")
        print(f"Rate limiting: {args.rate_limit} requests per {args.rate_window}s window")
        print(f"Maximum concurrent clients: {args.concurrent_clients}")
        
        if args.batch_mode:
            print(f"Running in batch mode with batch size: {args.batch_size}")
        
        # Run the benchmark using asyncio
        asyncio.run(run_benchmark(args))
        
        print("Benchmark completed successfully")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())