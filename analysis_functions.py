import pandas as pd
import numpy as np
import os
import json
import re
from src import get_model_shortname, GGB_Statements  # Assuming this is available and correctly defined

########################################################
# ANALYSIS HELPERS : TODO!
########################################################

def get_agent_shortname(agent_name_str):
    if not isinstance(agent_name_str, str):
        return 'unknown'
    
    # Attempt to extract model name if "agent_" prefix and "_<number>" suffix are present
    if agent_name_str.startswith("agent_") and '_' in agent_name_str[6:]:
        parts = agent_name_str[6:].split('_')
        # Check if the last part is a number (index)
        if parts[-1].isdigit():
            # Model name is everything between "agent_" and the last "_<number>"
            model_name_part = "_".join(parts[:-1])
            # Further simplify common model provider prefixes
            if model_name_part.startswith("openai_"):
                return model_name_part.replace("openai_", "gpt-")
            if model_name_part.startswith("anthropic_"):
                return model_name_part.replace("anthropic_", "")
            if model_name_part.startswith("google_"):
                return model_name_part.replace("google_", "")
            if model_name_part.startswith("mistralai_"):
                return model_name_part.replace("mistralai_", "")
            if model_name_part.startswith("cohere_"):
                return model_name_part.replace("cohere_", "")
            # Add more providers as needed
            return model_name_part # Return the extracted model name part
    
    # Fallback for other formats or if the above pattern doesn't match
    if 'gpt-4' in agent_name_str:
        return 'gpt-4'
    if 'gpt-3.5' in agent_name_str:
        return 'gpt-3.5'
    if 'claude-3-opus' in agent_name_str:
        return 'claude-3-opus'
    if 'claude-3-sonnet' in agent_name_str:
        return 'claude-3-sonnet'
    if 'claude-3-haiku' in agent_name_str:
        return 'claude-3-haiku'
    if 'gemini-1.5-pro' in agent_name_str:
        return 'gemini-1.5-pro'
    if 'gemini-1.0-pro' in agent_name_str: # Adjusted to match potential naming
        return 'gemini-1.0-pro'
    if 'mistral-large' in agent_name_str:
        return 'mistral-large'
    if 'command-r-plus' in agent_name_str:
        return 'command-r-plus'
    # Add more specific model name checks if needed
    
    return agent_name_str # Return original if no specific pattern matched


def load_and_clean_single_run(runcsv_list, Qs, label):
    """
    Loads and cleans data from single agent run CSV files.
    Filters out responses marked as off-topic based on a corresponding classification JSONL file.
    """
    main_df = pd.DataFrame()
    for runcsv_path_item in runcsv_list:
        try:
            temp_df = pd.read_csv(runcsv_path_item)
        except FileNotFoundError:
            print(f"Warning: CSV file not found: {runcsv_path_item}. Skipping.")
            continue

        temp_df['run_label'] = label
        temp_df['question_num'] = temp_df['question_id'].apply(lambda x: Qs.get_question_num(x))
        temp_df['category'] = temp_df['question_id'].apply(lambda x: Qs.get_category(x))
        
        # Attempt to load and merge classification data
        classification_jsonl_path = runcsv_path_item.replace('.csv', '_classification.jsonl')
        if os.path.exists(classification_jsonl_path):
            try:
                df_classified_single = pd.read_json(classification_jsonl_path, lines=True)
                
                # Ensure necessary columns exist for merging and filtering
                required_cols = ['question_id', 'run_index', 'model_name', 'is_response_off_topic']
                if not all(col in df_classified_single.columns for col in required_cols):
                    print(f"Warning: Classification file {classification_jsonl_path} is missing one or more required columns: {required_cols}. Skipping off-topic filtering for this file.")
                else:
                    # Prepare for merge: model_name in temp_df is raw, ensure df_classified_single matches.
                    # We assume grade.py stores the raw model_name from the input CSV for single runs.
                    merged_df = pd.merge(temp_df, 
                                         df_classified_single[required_cols],
                                         on=['question_id', 'run_index', 'model_name'], # Assumes 'model_name' is raw in both
                                         how='left')
                    
                    if 'is_response_off_topic' in merged_df.columns:
                        off_topic_mask = merged_df['is_response_off_topic'] == True
                        temp_df.loc[off_topic_mask, 'answer'] = np.nan
                        print(f"Filtered {off_topic_mask.sum()} off-topic responses from {runcsv_path_item} based on {classification_jsonl_path}.")
                    # No 'else' needed here as the all-cols-check handles missing 'is_response_off_topic' earlier
            except ValueError as e:
                print(f"Warning: Could not parse classification file {classification_jsonl_path}. Error: {e}. Proceeding without filtering for this file.")
            except Exception as e:
                print(f"Warning: An unexpected error occurred while processing classification file {classification_jsonl_path}. Error: {e}. Proceeding without filtering for this file.")
        else:
            print(f"Warning: Classification file not found: {classification_jsonl_path}. Proceeding without filtering off-topic responses for this file.")
            
        main_df = pd.concat([main_df, temp_df], ignore_index=True)
    return main_df

def ring_csv_to_df(csv_file, current_Qs):
    """
    Processes a single ring experiment CSV file into a structured DataFrame.
    Filters out agent messages marked as off-topic based on a corresponding classification JSONL file.
    """
    try:
        df_raw = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Warning: Ring CSV file not found: {csv_file}. Returning empty DataFrame.")
        return pd.DataFrame()

    classification_jsonl_path = csv_file.replace('.csv', '_classification.jsonl')
    df_classified_ring = None
    if os.path.exists(classification_jsonl_path):
        try:
            df_classified_ring = pd.read_json(classification_jsonl_path, lines=True)
            # Expected columns in df_classified_ring:
            # 'question_id', 'source_conversation_run_index', 'agent_name', 
            # 'message_index', 'is_response_off_topic'
            required_cols = ['question_id', 'source_conversation_run_index', 'agent_name', 'message_index', 'is_response_off_topic']
            if not all(col in df_classified_ring.columns for col in required_cols):
                print(f"Warning: Classification file {classification_jsonl_path} is missing one or more required columns: {required_cols}. Off-topic filtering will be skipped.")
                df_classified_ring = None # Disable filtering
        except ValueError as e:
            print(f"Warning: Could not parse classification file {classification_jsonl_path}. Error: {e}. Off-topic filtering will be skipped.")
            df_classified_ring = None
        except Exception as e:
            print(f"Warning: An unexpected error occurred while processing classification file {classification_jsonl_path}. Error: {e}. Off-topic filtering will be skipped.")
            df_classified_ring = None
    else:
        print(f"Warning: Classification file not found: {classification_jsonl_path}. Proceeding without filtering off-topic responses.")

    data_for_df = []
    for idx, row in df_raw.iterrows():
        conv_run_index = row['run_index']
        q_id = row['question_id']
        chat_type = row['chat_type']
        config_details = json.loads(row['config_details']) if isinstance(row['config_details'], str) else row['config_details']
        
        try:
            agent_responses_list = json.loads(row['agent_responses'])
        except json.JSONDecodeError:
            print(f"Warning: Could not parse agent_responses for q_id {q_id}, run_index {conv_run_index} in {csv_file}. Skipping this row.")
            continue

        for agent_msg in agent_responses_list:
            agent_name = agent_msg.get('agent_name')
            message_index_from_msg = agent_msg.get('message_index')
            
            is_off_topic = False
            if df_classified_ring is not None:
                mask = (
                    (df_classified_ring['question_id'] == q_id) &
                    (df_classified_ring['source_conversation_run_index'] == conv_run_index) &
                    (df_classified_ring['agent_name'] == agent_name) &
                    (df_classified_ring['message_index'] == message_index_from_msg)
                )
                classified_entry = df_classified_ring[mask]
                if not classified_entry.empty:
                    is_off_topic = classified_entry['is_response_off_topic'].iloc[0]
                # else: No classification found for this specific message, treat as not off-topic or log warning.
                # For now, if no entry, it defaults to is_off_topic = False.

            current_answer_val = agent_msg.get('answer')
            # Ensure answer is numeric, otherwise NaN. grade.py should put numeric answer in 'extracted_answer'.
            # Here, 'answer' is from the original CSV's agent_responses.
            try:
                numeric_answer = float(current_answer_val) if current_answer_val is not None else np.nan
            except (ValueError, TypeError):
                numeric_answer = np.nan

            if is_off_topic:
                numeric_answer = np.nan
            
            data_for_df.append({
                'question_id': q_id,
                'question_num': current_Qs.get_question_num(q_id),
                'category': current_Qs.get_category(q_id),
                'run_index': conv_run_index,
                'chat_type': chat_type,
                'config_details': config_details,
                'round_num': agent_msg.get('round_num', (message_index_from_msg // len(config_details.get('agent_names', [None]))) + 1 if message_index_from_msg is not None and config_details.get('agent_names') else np.nan), # Approximate round
                'agent_name': agent_name,
                'agent_answer': numeric_answer,
                'agent_confidence': agent_msg.get('confidence', np.nan),
                'full_response': agent_msg.get('message_content', ''),
                'message_index': message_index_from_msg
            })
            
    if not data_for_df:
        return pd.DataFrame()
    return pd.DataFrame(data_for_df)

def ring_to_roundrobin_df(df, current_Qs):
    """
    Converts a DataFrame from ring_csv_to_df to a round-robin format.
    Assumes df already has potentially NaN answers for off-topic responses.
    """
    if df.empty:
        return pd.DataFrame()

    round_robin_data = []
    
    # Group by each conversation instance (question_id and run_index)
    for name, group in df.groupby(['question_id', 'run_index', 'chat_type']):
        q_id, run_idx, chat_type_val = name
        
        # Sort by message_index to maintain order within the conversation
        group = group.sort_values(by='message_index')
        
        num_agents_in_convo = len(group['agent_name'].unique()) # Approx; better from config_details if available
        if 'config_details' in group.columns and isinstance(group['config_details'].iloc[0], dict):
            num_agents_in_convo = len(group['config_details'].iloc[0].get('agent_names', []))
        
        if num_agents_in_convo == 0: continue

        for i, row in group.iterrows():
            round_robin_data.append({
                'question_id': q_id,
                'question_num': row['question_num'],
                'category': row['category'],
                'run_index': run_idx, # This is the original conversation run_index
                'chat_type': chat_type_val,
                'config_details': row['config_details'],
                'round': (row['message_index'] // num_agents_in_convo) + 1 if row['message_index'] is not None else row.get('round_num', np.nan),
                'agent_name': row['agent_name'],
                'agent_answer': row['agent_answer'], # This will be NaN if filtered as off-topic
                'agent_confidence': row['agent_confidence'],
                'full_response': row['full_response'],
                'message_index': row['message_index'], # Keep original message index for reference
                'repeat_index': run_idx # In GGB_Analysis_new, repeat_index seems to be run_index
            })
            
    if not round_robin_data:
        return pd.DataFrame()
    return pd.DataFrame(round_robin_data)

def star_csv_to_df(csv_file, current_Qs, label_for_runtype="star"):
    """
    Processes a single star experiment CSV file into a structured DataFrame.
    Filters out agent messages marked as off-topic based on a corresponding classification JSONL file.
    (Placeholder implementation for structure - details depend on star CSV format)
    """
    try:
        df_raw = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Warning: Star CSV file not found: {csv_file}. Returning empty DataFrame.")
        return pd.DataFrame()

    classification_jsonl_path = csv_file.replace('.csv', '_classification.jsonl')
    df_classified_star = None
    if os.path.exists(classification_jsonl_path):
        try:
            df_classified_star = pd.read_json(classification_jsonl_path, lines=True)
            required_cols = ['question_id', 'source_conversation_run_index', 'agent_name', 'message_index', 'is_response_off_topic']
            if not all(col in df_classified_star.columns for col in required_cols):
                print(f"Warning: Classification file {classification_jsonl_path} is missing one or more required columns: {required_cols}. Off-topic filtering will be skipped.")
                df_classified_star = None
        except ValueError as e:
            print(f"Warning: Could not parse classification file {classification_jsonl_path}. Error: {e}. Off-topic filtering will be skipped.")
            df_classified_star = None
        except Exception as e:
            print(f"Warning: An unexpected error occurred while processing classification file {classification_jsonl_path}. Error: {e}. Off-topic filtering will be skipped.")
            df_classified_star = None
    else:
        print(f"Warning: Classification file not found: {classification_jsonl_path}. Proceeding without filtering off-topic responses.")

    data_for_df = []
    # This loop structure is a placeholder and needs to be adapted to the actual star CSV format
    for idx, row in df_raw.iterrows():
        conv_run_index = row.get('run_index', idx) # Or however conversation instance is identified
        q_id = row['question_id']
        # chat_type = row.get('chat_type', label_for_runtype) # Or from CSV
        # Assuming 'agent_responses' is a JSON string similar to ring topology
        
        try:
            agent_responses_list = json.loads(row.get('agent_responses', '[]'))
        except json.JSONDecodeError:
            print(f"Warning: Could not parse agent_responses for q_id {q_id}, run_index {conv_run_index} in {csv_file}. Skipping this row.")
            continue

        for agent_msg in agent_responses_list:
            agent_name = agent_msg.get('agent_name')
            message_index_from_msg = agent_msg.get('message_index') # Crucial for matching
            
            is_off_topic = False
            if df_classified_star is not None:
                mask = (
                    (df_classified_star['question_id'] == q_id) &
                    (df_classified_star['source_conversation_run_index'] == conv_run_index) &
                    (df_classified_star['agent_name'] == agent_name) &
                    (df_classified_star['message_index'] == message_index_from_msg)
                )
                classified_entry = df_classified_star[mask]
                if not classified_entry.empty:
                    is_off_topic = classified_entry['is_response_off_topic'].iloc[0]

            current_answer_val = agent_msg.get('answer')
            try:
                numeric_answer = float(current_answer_val) if current_answer_val is not None else np.nan
            except (ValueError, TypeError):
                numeric_answer = np.nan

            if is_off_topic:
                numeric_answer = np.nan

            # Add other relevant fields from star CSV structure
            data_for_df.append({
                'question_id': q_id,
                'question_num': current_Qs.get_question_num(q_id),
                'category': current_Qs.get_category(q_id),
                'run_index': conv_run_index, # Conversation instance identifier
                'chat_type': row.get('chat_type', label_for_runtype), # Get from row or use default
                'config_details': json.loads(row['config_details']) if isinstance(row.get('config_details'), str) else row.get('config_details'),
                'agent_name': agent_name,
                'agent_answer': numeric_answer, # Will be NaN if filtered
                'agent_confidence': agent_msg.get('confidence', np.nan),
                'full_response': agent_msg.get('message_content', ''),
                'message_index': message_index_from_msg,
                # Add star-specific fields like 'is_supervisor', 'loop_num' etc.
                'is_supervisor': agent_msg.get('is_supervisor', False), # Example
                'loop_num': agent_msg.get('loop_num', np.nan) # Example
            })
            
    if not data_for_df:
        return pd.DataFrame()
    return pd.DataFrame(data_for_df)