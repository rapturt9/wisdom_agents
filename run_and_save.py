import os
import csv
import asyncio
import json
from datetime import datetime
from helpers import get_prompt, run_single_agent_chat, extract_answer_from_response, extract_confidence_from_response

# Your existing functions remain unchanged
# prompt = get_prompt(group_chat=False)
# async def run_single_agent_chat(question_number = 1): ...
# def extract_answer_from_response(content): ...
# def extract_confidence_from_response(content): ...
# async def run_multiple_agents_chat(num_runs=100, question_number=0, model=model): ...

def get_checkpoint_filename(base_dir='checkpoints'):
    """Create a checkpoint filename based on the current timestamp."""
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"checkpoint_{timestamp}.json")


def save_checkpoint(completed_runs, checkpoint_file=None):
    """Save the current progress to a checkpoint file."""
    if checkpoint_file is None:
        checkpoint_file = get_checkpoint_filename()
    
    with open(checkpoint_file, 'w') as f:
        json.dump(completed_runs, f)
    
    print(f"Checkpoint saved to {checkpoint_file}")
    return checkpoint_file


def load_checkpoint(checkpoint_file):
    """Load progress from a checkpoint file."""
    if not os.path.exists(checkpoint_file):
        print(f"Checkpoint file {checkpoint_file} not found. Starting fresh.")
        return {}
    
    with open(checkpoint_file, 'r') as f:
        completed_runs = json.load(f)
    
    print(f"Loaded checkpoint from {checkpoint_file}, with {len(completed_runs)} completed runs.")
    return completed_runs


def write_to_csv_files(results, csv_file1, csv_file2):
    """Write results to two CSV files according to the specified format."""
    # Check if files exist already
    file1_exists = os.path.exists(csv_file1)
    file2_exists = os.path.exists(csv_file2)
    
    # Write to csv_file1: 'model_name, question_num, answer, confidence'
    with open(csv_file1, 'a', newline='') as f1:
        writer1 = csv.writer(f1)
        if not file1_exists:
            writer1.writerow(['model_name', 'question_num', 'answer', 'confidence'])
        
        for result in results:
            writer1.writerow([
                result['model_name'],
                result['question_num'],
                result['answer'],
                result['confidence']
            ])
    
    # Write to csv_file2: 'model_name, question_num, full_response'
    with open(csv_file2, 'a', newline='') as f2:
        writer2 = csv.writer(f2)
        if not file2_exists:
            writer2.writerow(['model_name', 'question_num', 'full_response'])
        
        for result in results:
            writer2.writerow([
                result['model_name'],
                result['question_num'],
                result['full_response']
            ])


async def main(models, question_range=(1, 88), num_runs=1, checkpoint_file=None):
    """Main function to run the evaluation across all models and questions."""
    # If no checkpoint file is specified, create a new one
    if checkpoint_file is None:
        checkpoint_file = get_checkpoint_filename()
        completed_runs = {}
    else:
        completed_runs = load_checkpoint(checkpoint_file)
    
    # Setup output files
    csv_file1 = "agent_answers.csv"
    csv_file2 = "agent_full_responses.csv"
    
    all_results = []
    question_numbers = list(range(question_range[0], question_range[1] + 1))
    
    # Process each model
    for model_name in models:
        print(f"Processing model: {model_name}")
        
        # Get completed questions for this model from checkpoint
        model_key = str(model_name)
        if model_key not in completed_runs:
            completed_runs[model_key] = {}
        
        # Process each question for this model
        for question_num in question_numbers:
            q_key = str(question_num)
            
            # Skip if already completed
            if q_key in completed_runs[model_key]:
                print(f"Skipping question {question_num} for model {model_name} (already completed)")
                continue
                
            try:
                print(f"Processing question {question_num} for model {model_name}")
                
                # Use your existing function to run multiple agents for this question
                responses = await run_multiple_agents_chat(
                    num_runs=num_runs,
                    question_number=question_num,
                    model=model_name
                )
                
                # Process each response
                model_results = []
                for i, response in enumerate(responses):
                    # Get full response - assuming your run_multiple_agents_chat returns
                    # the full response or we can adjust as needed
                    full_response = response
                    
                    # Extract answer and confidence
                    answer = extract_answer_from_response(full_response)
                    confidence = extract_confidence_from_response(full_response)
                    
                    # Create result object
                    result_obj = {
                        "model_name": model_name,
                        "question_num": question_num,
                        "answer": answer,
                        "confidence": confidence,
                        "full_response": full_response
                    }
                    
                    model_results.append(result_obj)
                
                # Write results to CSV
                write_to_csv_files(model_results, csv_file1, csv_file2)
                all_results.extend(model_results)
                
                # Mark as completed in checkpoint
                completed_runs[model_key][q_key] = True
                
                # Save checkpoint after each question
                save_checkpoint(completed_runs, checkpoint_file)
                
            except Exception as e:
                print(f"Error processing question {question_num} with model {model_name}: {str(e)}")
        
        # Save checkpoint after each model
        save_checkpoint(completed_runs, checkpoint_file)
    
    print(f"All runs completed. Processed {len(all_results)} questions across {len(models)} models.")
    print(f"Results saved to {csv_file1} and {csv_file2}")


if __name__ == "__main__":
    # Example usage:
    models = ["model_name_1", "model_name_2"]  # Replace with your actual model names
    
    # Optional: Load from a specific checkpoint file
    checkpoint_file = None  # or specify like "checkpoints/checkpoint_20250430_120000.json"
    
    # Run the evaluation
    asyncio.run(main(
        models=models,
        question_range=(1, 88),  # Questions from 1 to 88
        num_runs=1,             # Number of runs per question (adjust as needed)
        checkpoint_file=checkpoint_file
    ))