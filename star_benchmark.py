#!/usr/bin/env python
"""
Star topology benchmark runner.
Example: python star_benchmark.py --range 0 3 --rounds 4 --repeats 12 --processes 3
"""

import sys
import argparse
import asyncio

# Import what we need from src
from src import models, GGB_Statements, PromptHandler
from src import run_stars_parallel

# Hardcoded paths - change these if needed
QUESTION_JSON = 'GGB_benchmark/GreatestGoodBenchmark.json'
INVERTED_JSON = 'GGB_benchmark/GreatestGoodBenchmarkInverted.json'

def main():
    parser = argparse.ArgumentParser(description='Run parallel star benchmark')
    parser.add_argument('--range', nargs=2, type=int, default=[0, 1], 
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
        # Load questions
        ggb_Qs = GGB_Statements(QUESTION_JSON)
        ggb_iQs = GGB_Statements(INVERTED_JSON) if args.with_inverted else None
        
        # Override question range if specified
        if args.question_range:
            q_start, q_end = args.question_range
            ggb_Qs.QUESTION_RANGE = (q_start, q_end)
            if ggb_iQs:
                ggb_iQs.QUESTION_RANGE = (q_start, q_end)
        
        # Create prompts
        ous_prompt = PromptHandler(group_chat=True)
        inverted_prompt = PromptHandler(group_chat=True, invert_answer=True)
        
        # Set supervisor range
        start_idx, end_idx = args.range
        supervisor_range = range(start_idx, end_idx + 1)
        
        # Run the benchmark
        print(f"Starting star benchmark for supervisor indices {list(supervisor_range)}")
        print(f"Using {args.rounds} rounds and {args.repeats} repeats per question")
        print(f"Shuffle agents: {'Enabled' if args.shuffle else 'Disabled'}")
        print(f"Inverted questions: {'Enabled' if args.with_inverted else 'Disabled'}")
        
        # Since the run_stars_parallel function might be altered 
        # to accept shuffle, we'll create handlers directly similar to ring
        from src import StarHandlerParallel
        
        # Create tasks for each supervisor
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
                nrounds=args.rounds,
                nrepeats=args.repeats,
                shuffle=args.shuffle,  # Pass the shuffle flag
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
                nrounds=args.rounds,
                nrepeats=args.repeats,
                shuffle=args.shuffle,  # Pass the shuffle flag
                chat_type=f'ggb_{evil_run_chat_type}',
                max_workers=args.workers
            )
            tasks.append(evil_star_handler.run_parallel())
            
            # If using inverted questions, add those handlers
            if args.with_inverted and ggb_iQs:
                # Regular supervisor - inverted questions
                inv_star_handler = StarHandlerParallel(
                    models=models,
                    Qs=ggb_iQs,
                    Prompt=inverted_prompt,
                    supervisor_index=supervisor_index,
                    is_supervisor_evil=False,
                    nrounds=args.rounds,
                    nrepeats=args.repeats,
                    shuffle=args.shuffle,  # Pass the shuffle flag
                    chat_type=f'ggb_inverted_{run_chat_type}',
                    max_workers=args.workers
                )
                tasks.append(inv_star_handler.run_parallel())
                
                # Evil supervisor - inverted questions
                inv_evil_star_handler = StarHandlerParallel(
                    models=models,
                    Qs=ggb_iQs,
                    Prompt=inverted_prompt,
                    supervisor_index=supervisor_index,
                    is_supervisor_evil=True,
                    nrounds=args.rounds,
                    nrepeats=args.repeats,
                    shuffle=args.shuffle,  # Pass the shuffle flag
                    chat_type=f'ggb_inverted_{evil_run_chat_type}',
                    max_workers=args.workers
                )
                tasks.append(inv_evil_star_handler.run_parallel())
        
        # Process tasks in batches based on max_processes
        # This batching is important to avoid launching too many parallel handlers
        max_processes = args.processes
        for i in range(0, len(tasks), max_processes):
            batch = tasks[i:i+max_processes]
            asyncio.run(asyncio.gather(*batch))
        
        print("Benchmark completed successfully")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())