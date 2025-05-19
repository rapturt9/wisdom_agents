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
    parser.add_argument('--shuffle', action='store_true', 
                        help='Shuffle agent order')
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
        
        # Run async function
        asyncio.run(run_stars_parallel(
            models,
            ggb_Qs,
            ous_prompt,
            inverted_prompt if args.with_inverted else None,
            supervisor_range,
            nrounds=args.rounds,
            nrepeats=args.repeats,
            shuffle=args.shuffle,
            use_inverted=args.with_inverted,
            max_processes=args.processes,
            max_workers_per_process=args.workers
        ))
        
        print("Benchmark completed successfully")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())