#!/usr/bin/env python
"""
Consolidate marker files into checkpoint files after benchmark runs.
Usage: python consolidate_markers.py
"""

import os
import glob
import json
import re
from collections import defaultdict
from filelock import FileLock

def load_checkpoint(checkpoint_file):
    """Load existing checkpoint data."""
    if not os.path.exists(checkpoint_file):
        return {}
    
    try:
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error loading {checkpoint_file}: {e}")
        return {}

def save_checkpoint(checkpoint_file, data):
    """Save checkpoint data."""
    try:
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        with open(checkpoint_file, 'w') as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving {checkpoint_file}: {e}")
        return False

def consolidate_markers():
    """Consolidate all marker files into their corresponding checkpoint files."""
    checkpoint_dir = 'checkpoints'
    
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} does not exist")
        return
    
    # Find all marker files
    pattern = os.path.join(checkpoint_dir, ".temp_*_*_*.marker")
    marker_files = glob.glob(pattern)
    
    if not marker_files:
        print("No marker files found - nothing to consolidate!")
        return
    
    print(f"Found {len(marker_files)} marker files to consolidate")
    
    # Group markers by chat_type to find corresponding checkpoint files
    markers_by_chat_type = defaultdict(list)
    
    for marker_file in marker_files:
        filename = os.path.basename(marker_file)
        
        # Extract info from filename: .temp_{chat_type}_{question_num}_{iteration_idx}.marker
        match = re.match(r"\.temp_([^_]+(?:_[^_]+)*)_(\d+)_(\d+)\.marker$", filename)
        if match:
            chat_type = match.group(1)
            question_num = match.group(2)
            iteration_idx = match.group(3)
            
            markers_by_chat_type[chat_type].append({
                'file': marker_file,
                'question_num': question_num,
                'iteration_idx': iteration_idx
            })
        else:
            print(f"Warning: Could not parse marker filename: {filename}")
    
    print(f"Found markers for {len(markers_by_chat_type)} chat types:")
    for chat_type, markers in markers_by_chat_type.items():
        print(f"  - {chat_type}: {len(markers)} markers")
    
    # Find corresponding checkpoint files and consolidate
    consolidated_count = 0
    
    for chat_type, markers in markers_by_chat_type.items():
        print(f"\nProcessing {chat_type}...")
        
        # Find the checkpoint file for this chat type
        checkpoint_pattern = os.path.join(checkpoint_dir, f"*{chat_type}*_checkpoint.json")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if not checkpoint_files:
            print(f"  Warning: No checkpoint file found for {chat_type}")
            print(f"  Looking for pattern: {checkpoint_pattern}")
            continue
        
        if len(checkpoint_files) > 1:
            print(f"  Warning: Multiple checkpoint files found for {chat_type}:")
            for cf in checkpoint_files:
                print(f"    - {os.path.basename(cf)}")
            print(f"  Using: {os.path.basename(checkpoint_files[0])}")
        
        checkpoint_file = checkpoint_files[0]
        print(f"  Using checkpoint: {os.path.basename(checkpoint_file)}")
        
        # Use file locking to safely update the checkpoint
        lock_file = f"{checkpoint_file}.lock"
        with FileLock(lock_file):
            # Load existing checkpoint data
            checkpoint_data = load_checkpoint(checkpoint_file)
            
            # Add marker data to checkpoint
            updates_made = 0
            for marker_info in markers:
                q_num = marker_info['question_num']
                iter_idx = marker_info['iteration_idx']
                
                if q_num not in checkpoint_data:
                    checkpoint_data[q_num] = {}
                
                if iter_idx not in checkpoint_data[q_num]:
                    checkpoint_data[q_num][iter_idx] = True
                    updates_made += 1
            
            # Save updated checkpoint
            if updates_made > 0:
                if save_checkpoint(checkpoint_file, checkpoint_data):
                    print(f"  Updated checkpoint with {updates_made} completed tasks")
                    consolidated_count += len(markers)
                else:
                    print(f"  Error saving checkpoint file")
            else:
                print(f"  No new updates needed (all tasks already in checkpoint)")
    
    print(f"\nConsolidation complete!")
    print(f"Consolidated {consolidated_count} marker files into checkpoint files")
    print(f"\nYou can now safely delete the marker files:")
    print(f"  find {checkpoint_dir} -name '.temp_*_*_*.marker' -delete")
    print(f"  or manually delete them from the {checkpoint_dir} directory")

if __name__ == "__main__":
    consolidate_markers()