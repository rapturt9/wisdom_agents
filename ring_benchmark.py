#!/usr/bin/env python
"""
Ring topology benchmark runner.
Example: python ring_benchmark.py --range 0 3 --rounds 4 --repeats 12 --shuffle
"""

import sys
import argparse
import asyncio

# Import what we need from src
from src import models, GGB_Statements, PromptHandler
from src import run_all_rings_parallel

# Hardcoded paths - change these if needed
QUESTION_JSON = 'GGB_benchmark/GreatestGoodBenchmark.json'
INVERTED_JSON = 'GGB_benchmark/GreatestGoodBenchmarkInverted.json'

def main():
    parser = argparse.ArgumentParser(description='Run parallel ring benchmark')
    parser.add_argument('--range', nargs=2, type=int, default=[0, 1], 
                        help='Range of model indices (start and end inclusive)')
    parser.add_argument('--rounds', type=int, default=4, 
                        help='Number of rounds in each conversation')
    parser.add_argument('--repeats', type=int, default=12, 
                        help='Number of iterations per question')
    parser.add_argument('--shuffle', action='store_true', 
                        help='Shuffle agent order')
    parser.add_argument('--with-inverted', action='store_true', 
                        help='Also run with inverted questions')
    parser.add_argument('--question-range', nargs=2, type=int, default=None,
                        help='Range of questions to run (start and end inclusive)')
    parser.add_argument('--workers', type=int, default=8,
                       help='Maximum number of workers per handler')
    
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
        
        # Set model range
        start_idx, end_idx = args.range
        my_range = range(start_idx, end_idx + 1)
        
        # Run the benchmark
        print(f"Starting ring benchmark for model indices {list(my_range)}")
        print(f"Using {args.rounds} rounds and {args.repeats} repeats per question")
        print(f"Shuffle agents: {'Enabled' if args.shuffle else 'Disabled'}")
        print(f"Inverted questions: {'Enabled' if args.with_inverted else 'Disabled'}")
        
        # We need to modify run_all_rings_parallel to accept the shuffle parameter
        # For now, let's directly create and run handlers one by one
        from src import RingHandlerParallel
        
        # Create tasks for each model configuration
        tasks = []
        n_models = len(models)
        
        for i in my_range:
            if i == 0:
                run_models = models
                run_chat_type = 'hetero_ring'
            else:
                run_models = [models[i-1]] * n_models
                model_shortname = models[i-1].split('/')[-1]
                run_chat_type = f'{model_shortname}_ring'
            
            # Regular questions
            ring = RingHandlerParallel(
                models=run_models,
                Qs=ggb_Qs,
                Prompt=ous_prompt,
                nrounds=args.rounds,
                nrepeats=args.repeats,
                shuffle=args.shuffle,
                chat_type=f'ggb_{run_chat_type}',
                max_workers=args.workers
            )
            tasks.append(ring.run_parallel())
            
            # Inverted questions if requested
            if args.with_inverted and ggb_iQs:
                inv_ring = RingHandlerParallel(
                    models=run_models,
                    Qs=ggb_iQs,
                    Prompt=inverted_prompt,
                    nrounds=args.rounds,
                    nrepeats=args.repeats,
                    shuffle=args.shuffle,
                    chat_type=f'ggb_inverted_{run_chat_type}',
                    max_workers=args.workers
                )
                tasks.append(inv_ring.run_parallel())
        
        # Run all tasks concurrently
        asyncio.run(asyncio.gather(*tasks))
        
        print("Benchmark completed successfully")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())