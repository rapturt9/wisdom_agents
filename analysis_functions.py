import pandas as pd
import numpy as np
import os
import json
import re
from src import get_model_shortname, GGB_Statements  # Assuming this is available and correctly defined

########################################################
# ANALYSIS HELPERS : TODO!
########################################################

def get_agent_shortname(agent_name):
    short_names = ['claude', 'gpt', 'deepseek', 'llama', 'gemini', 'qwen']
    name = [n for n in short_names if n in agent_name]
    if len(name) > 1:
        print(f'cannot resolve name for {agent_name}. please fix.')
        return
    return name[0]


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

        # temp_df already has a question_num
        temp_df['category'] = temp_df['question_id'].apply(lambda x: Qs.get_question_category(x))

        # Initialize classification columns with default values
        temp_df['selected_categories'] = None
        temp_df['is_response_off_topic'] = False
        temp_df['off_topic_reason'] = None
        
        # Attempt to load and merge classification data
        classification_jsonl_path = runcsv_path_item.replace('.csv', '_classification.jsonl')
        if os.path.exists(classification_jsonl_path):
            try:
                df_classified_single = pd.read_json(classification_jsonl_path, lines=True)
                
                # Ensure necessary columns exist for merging and filtering
                required_cols = ['question_id', 'run_index', 'model_name', 'is_response_off_topic']
                classification_cols = ['selected_categories', 'is_response_off_topic', 'off_topic_reason']
                
                if not all(col in df_classified_single.columns for col in required_cols):
                    print(f"Warning: Classification file {classification_jsonl_path} is missing one or more required columns: {required_cols}. Skipping off-topic filtering for this file.")
                else:
                    # Convert selected_categories list to comma-separated string if it's a list
                    if 'selected_categories' in df_classified_single.columns:
                        df_classified_single['selected_categories_str'] = df_classified_single['selected_categories'].apply(
                            lambda x: ','.join(x) if isinstance(x, list) else str(x) if x is not None else None
                        )
                    else:
                        df_classified_single['selected_categories_str'] = None
                    
                    # Prepare columns for merge
                    merge_cols = ['question_id', 'run_index', 'model_name', 'selected_categories_str', 'is_response_off_topic', 'off_topic_reason']
                    available_merge_cols = [col for col in merge_cols if col in df_classified_single.columns or col == 'selected_categories_str']
                    
                    merged_df = pd.merge(temp_df, 
                                         df_classified_single[available_merge_cols],
                                         on=['question_id', 'run_index', 'model_name'],
                                         how='left',
                                         suffixes=('', '_classified'))
                    
                    # Update classification columns
                    if 'selected_categories_str' in merged_df.columns:
                        temp_df['selected_categories'] = merged_df['selected_categories_str']
                    if 'is_response_off_topic_classified' in merged_df.columns:
                        temp_df['is_response_off_topic'] = merged_df['is_response_off_topic_classified'].fillna(False)
                        # Filter out off-topic responses
                        off_topic_mask = temp_df['is_response_off_topic'] == True
                        temp_df.loc[off_topic_mask, 'answer'] = np.nan
                        print(f"Filtered {off_topic_mask.sum()} off-topic responses from {runcsv_path_item} based on {classification_jsonl_path}.")
                    if 'off_topic_reason_classified' in merged_df.columns:
                        temp_df['off_topic_reason'] = merged_df['off_topic_reason_classified']
                        
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
        print(f"  Loaded CSV with {len(df_raw)} rows and columns: {df_raw.columns.tolist()}")
    except FileNotFoundError:
        print(f"Warning: Ring CSV file not found: {csv_file}. Returning empty DataFrame.")
        return pd.DataFrame()

    if df_raw.empty:
        print(f"Warning: CSV file {csv_file} is empty.")
        return pd.DataFrame()

    # Debug: Check what columns we actually have
    print(f"  CSV columns: {df_raw.columns.tolist()}")
    print(f"  Sample row keys: {df_raw.iloc[0].to_dict().keys() if len(df_raw) > 0 else 'No rows'}")

    classification_jsonl_path = csv_file.replace('.csv', '_classification.jsonl')
    df_classified_ring = None
    if os.path.exists(classification_jsonl_path):
        try:
            df_classified_ring = pd.read_json(classification_jsonl_path, lines=True)
            required_cols = ['question_id', 'question_num', 'agent_name', 'message_index', 'is_response_off_topic']
            if not all(col in df_classified_ring.columns for col in required_cols):
                print(f"Warning: Classification file {classification_jsonl_path} is missing one or more required columns: {required_cols}. Off-topic filtering will be skipped.")
                df_classified_ring = None
            else:
                # Convert selected_categories list to comma-separated string if it's a list
                if 'selected_categories' in df_classified_ring.columns:
                    df_classified_ring['selected_categories_str'] = df_classified_ring['selected_categories'].apply(
                        lambda x: ','.join(x) if isinstance(x, list) else str(x) if x is not None else None
                    )
        except Exception as e:
            print(f"Warning: Error processing classification file {classification_jsonl_path}: {e}")
            df_classified_ring = None


    data_for_df = []
    
    # Process each row in the CSV
    for idx, row in df_raw.iterrows():
        conv_run_index = row.get('run_index', idx)
        q_num = row.get('question_num')
        chat_type = row.get('chat_type', 'unknown')
        q_num = row.get('question_num')
        
        # Handle config_details
        config_details = row.get('config_details', {})
        if isinstance(config_details, str):
            try:
                config_details = json.loads(config_details)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse config_details for row {idx}")
                config_details = {}
        
        # Handle agent_responses - this is the key field that contains the actual responses
        agent_responses_raw = row.get('agent_responses', '[]')
        if pd.isna(agent_responses_raw):
            print(f"Warning: No agent_responses for q_id {q_id}, run_index {conv_run_index}")
            continue
            
        try:
            if isinstance(agent_responses_raw, str):
                agent_responses_list = json.loads(agent_responses_raw)
            else:
                agent_responses_list = agent_responses_raw
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse agent_responses for q_id {q_id}, run_index {conv_run_index}: {e}")
            continue

        if not agent_responses_list:
            print(f"Warning: Empty agent_responses for q_id {q_id}, run_index {conv_run_index}")
            continue

        # Process each agent message
        for agent_msg in agent_responses_list:
            agent_name = agent_msg.get('agent_name', 'unknown')
            message_index_from_msg = agent_msg.get('message_index')
            
            # Initialize classification fields
            is_off_topic = False
            selected_categories_str = None
            off_topic_reason = None
            
            # Try to get classification data
            if df_classified_ring is not None:
                mask = (
                    (df_classified_ring['question_id'] == q_id) &
                    (df_classified_ring['agent_name'] == agent_name) &
                    (df_classified_ring['message_index'] == message_index_from_msg)
                )
                
                if 'source_conversation_run_index' in df_classified_ring.columns:
                    mask = mask & (df_classified_ring['source_conversation_run_index'] == conv_run_index)
                
                classified_entry = df_classified_ring[mask]
                if not classified_entry.empty:
                    is_off_topic = classified_entry['is_response_off_topic'].iloc[0]
                    if 'selected_categories_str' in classified_entry.columns:
                        selected_categories_str = classified_entry['selected_categories_str'].iloc[0]
                    if 'off_topic_reason' in classified_entry.columns:
                        off_topic_reason = classified_entry['off_topic_reason'].iloc[0]

            # Extract numeric answer
            current_answer_val = agent_msg.get('answer')
            try:
                numeric_answer = float(current_answer_val) if current_answer_val is not None else np.nan
            except (ValueError, TypeError):
                numeric_answer = np.nan

            # Filter out off-topic responses
            if is_off_topic:
                numeric_answer = np.nan
            
            
            # Determine round number
            if isinstance(config_details, dict) and 'ensemble' in config_details:
                num_agents = sum(entry.get('number', 1) for entry in config_details['ensemble'])
            else:
                # Fallback: count unique agent names in this conversation
                num_agents = len(set(msg.get('agent_name', '') for msg in agent_responses_list))
            
            round_num = (message_index_from_msg // num_agents) + 1 if message_index_from_msg is not None and num_agents > 0 else 1
            
            data_for_df.append({
                'question_id': q_id,
                'question_num': q_num,
                'category': current_Qs.get_question_category(q_id),
                'run_index': conv_run_index,
                'chat_type': chat_type,
                'config_details': config_details,
                'round_num': round_num,
                'agent_name': agent_name,
                'agent_answer': numeric_answer,
                'agent_confidence': agent_msg.get('confidence', np.nan),
                'full_response': agent_msg.get('message_content', ''),
                'message_index': message_index_from_msg,
                'selected_categories': selected_categories_str,
                'is_response_off_topic': is_off_topic,
                'off_topic_reason': off_topic_reason
            })
            
    if not data_for_df:
        print(f"Warning: No valid data extracted from {csv_file}")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(data_for_df)
    print(f"  Successfully extracted {len(result_df)} agent responses")
    return result_df

def ring_to_roundrobin_df(df, current_Qs):
    """
    Converts a DataFrame from ring_csv_to_df to a round-robin format.
    Assumes df already has potentially NaN answers for off-topic responses.
    """
    if df.empty:
        return pd.DataFrame()

    round_robin_data = []
    
    # Group by each conversation instance (question_id and run_index)
    for name, group in df.groupby(['question_id', 'question_num', 'run_index', 'chat_type']):
        q_id, run_idx, chat_type_val = name
        
        # Sort by message_index to maintain order within the conversation
        group = group.sort_values(by='message_index')
        
        # Get number of agents from config_details
        config_details = group['config_details'].iloc[0] if len(group) > 0 else {}
        if isinstance(config_details, dict) and 'ensemble' in config_details:
            # Count total agents across all ensemble entries
            num_agents_in_convo = sum(entry.get('number', 1) for entry in config_details['ensemble'])
        else:
            num_agents_in_convo = len(group['agent_name'].unique())
        
        if num_agents_in_convo == 0: 
            continue

        for i, row in group.iterrows():
            # Extract numeric answer from full_response using regex
            numeric_answer = row['agent_answer']  # This might be NaN from ring_csv_to_df
            
            if pd.isna(numeric_answer):
                # Try to extract from full_response if agent_answer is NaN
                full_response = str(row.get('full_response', ''))
                answer_match = re.search(r'<ANSWER>(\d+(?:\.\d+)?)</ANSWER>', full_response)
                if answer_match:
                    try:
                        numeric_answer = float(answer_match.group(1))
                    except (ValueError, TypeError):
                        numeric_answer = np.nan
                else:
                    numeric_answer = np.nan
            
            # Extract confidence similarly
            confidence_val = row['agent_confidence']  # This might be NaN
            if pd.isna(confidence_val):
                full_response = str(row.get('full_response', ''))
                conf_match = re.search(r'<CONF>(\d+(?:\.\d+)?)</CONF>', full_response)
                if conf_match:
                    try:
                        confidence_val = float(conf_match.group(1))
                    except (ValueError, TypeError):
                        confidence_val = np.nan
                else:
                    confidence_val = np.nan
            
            # Calculate round number
            round_num = (row['message_index'] // num_agents_in_convo) + 1 if row['message_index'] is not None else row.get('round_num', 1)
            
            # Calculate ggb_question_id (modulo 100 of question_id)
            ggb_question_id = q_id % 100 if q_id is not None else None
            
            round_robin_data.append({
                'question_id': q_id,
                'question_num': row['question_num'],
                'category': current_Qs.get_question_category(q_id),
                'run_index': run_idx,
                'chat_type': chat_type_val,
                'config_details': row['config_details'],
                'round': round_num,
                'agent_name': row['agent_name'],
                'agent_answer': numeric_answer,
                'agent_confidence': confidence_val,
                'full_response': row['full_response'],
                'message_index': row['message_index'],
                'repeat_index': run_idx,
                'ggb_question_id': ggb_question_id,  # Add this column
                'selected_categories': row.get('selected_categories'),
                'is_response_off_topic': row.get('is_response_off_topic', False),
                'off_topic_reason': row.get('off_topic_reason')
            })
            
    if not round_robin_data:
        return pd.DataFrame()
    return pd.DataFrame(round_robin_data)

def star_csv_to_df(csv_file, current_Qs, label_for_runtype="star"):
    """
    Processes a single star experiment CSV file into a structured DataFrame.
    Filters out agent messages marked as off-topic based on a corresponding classification JSONL file.
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
            required_cols = ['question_id','question_num', 'agent_name', 'message_index', 'is_response_off_topic']
            if not all(col in df_classified_star.columns for col in required_cols):
                print(f"Warning: Classification file {classification_jsonl_path} is missing one or more required columns: {required_cols}. Off-topic filtering will be skipped.")
                df_classified_star = None
            else:
                # Convert selected_categories list to comma-separated string if it's a list
                if 'selected_categories' in df_classified_star.columns:
                    df_classified_star['selected_categories_str'] = df_classified_star['selected_categories'].apply(
                        lambda x: ','.join(x) if isinstance(x, list) else str(x) if x is not None else None
                    )
        except ValueError as e:
            print(f"Warning: Could not parse classification file {classification_jsonl_path}. Error: {e}. Off-topic filtering will be skipped.")
            df_classified_star = None
        except Exception as e:
            print(f"Warning: An unexpected error occurred while processing classification file {classification_jsonl_path}. Error: {e}. Off-topic filtering will be skipped.")
            df_classified_star = None
    else:
        print(f"Warning: Classification file not found: {classification_jsonl_path}. Proceeding without filtering off-topic responses.")


    data_for_df = []
    for idx, row in df_raw.iterrows():
        conv_run_index = row.get('run_index', idx)
        q_id = row['question_id']
        q_num = row['question_num']
        
        try:
            agent_responses_list = json.loads(row.get('agent_responses', '[]'))
        except json.JSONDecodeError:
            print(f"Warning: Could not parse agent_responses for q_id {q_id}, run_index {conv_run_index} in {csv_file}. Skipping this row.")
            continue

        for agent_msg in agent_responses_list:
            agent_name = agent_msg.get('agent_name')
            message_index_from_msg = agent_msg.get('message_index')
            
            # Initialize classification fields
            is_off_topic = False
            selected_categories_str = None
            off_topic_reason = None
            
            if df_classified_star is not None:
                # Try different matching strategies for run_index
                mask = (
                    (df_classified_star['question_id'] == q_id) &
                    (df_classified_star['agent_name'] == agent_name) &
                    (df_classified_star['message_index'] == message_index_from_msg)
                )
                
                # If source_conversation_run_index exists, add it to the mask
                if 'source_conversation_run_index' in df_classified_star.columns:
                    mask = mask & (df_classified_star['source_conversation_run_index'] == conv_run_index)
                
                classified_entry = df_classified_star[mask]
                if not classified_entry.empty:
                    is_off_topic = classified_entry['is_response_off_topic'].iloc[0]
                    if 'selected_categories_str' in classified_entry.columns:
                        selected_categories_str = classified_entry['selected_categories_str'].iloc[0]
                    if 'off_topic_reason' in classified_entry.columns:
                        off_topic_reason = classified_entry['off_topic_reason'].iloc[0]

            current_answer_val = agent_msg.get('answer')
            try:
                numeric_answer = float(current_answer_val) if current_answer_val is not None else np.nan
            except (ValueError, TypeError):
                numeric_answer = np.nan

            if is_off_topic:
                numeric_answer = np.nan
                
            
            data_for_df.append({
                'question_id': q_id,
                'question_num': q_num,
                'category': current_Qs.get_question_category(q_id),
                'run_index': conv_run_index,
                'chat_type': row.get('chat_type', label_for_runtype),
                'config_details': json.loads(row['config_details']) if isinstance(row.get('config_details'), str) else row.get('config_details'),
                'agent_name': agent_name,
                'agent_answer': numeric_answer,
                'agent_confidence': agent_msg.get('confidence', np.nan),
                'full_response': agent_msg.get('message_content', ''),
                'message_index': message_index_from_msg,
                'is_supervisor': agent_msg.get('is_supervisor', False),
                'loop_num': agent_msg.get('loop_num', np.nan),
                'selected_categories': selected_categories_str,
                'is_response_off_topic': is_off_topic,
                'off_topic_reason': off_topic_reason
            })
            
    if not data_for_df:
        return pd.DataFrame()
    return pd.DataFrame(data_for_df)

def load_datasets_with_fallback(base_dir):
    """
    Discover all results_* directories under base_dir and return two dicts:
      - datasets[name] -> raw DataFrame placeholder
      - dataset_info[name] -> metadata dict
    """
    from pathlib import Path
    datasets = {}
    dataset_info = {}
    for d in Path(base_dir).glob("results*"):
        if not d.is_dir(): 
            continue
        name = d.name
        # placeholder empty frame; replace with actual loader (e.g. load_and_clean_single_run)
        datasets[name] = pd.DataFrame()
        dataset_info[name] = {
            "has_classification": False,
            "file_type": "unknown",
            "file_count": len(list(d.glob("*.csv")))
        }
    return datasets, dataset_info

def prepare_datasets_for_analysis(datasets, dataset_info):
    """
    Wrap each raw DataFrame into the standard dict with keys:
      analysis, exploded, errors, info
    """
    prepared = {}
    for name, df in datasets.items():
        prepared[name] = {
            "analysis": df,
            "exploded": df,            # replace with real exploded view
            "errors": pd.DataFrame(),  # replace with real errors
            "info": dataset_info.get(name, {})
        }
    return prepared

def create_combined_dataset(prepared_datasets):
    """
    Concatenate all prepared analyses/exploded/errors into a combined view.
    """
    combined_analysis = pd.concat(
        [v["analysis"] for v in prepared_datasets.values()],
        ignore_index=True
    )
    combined_exploded = pd.concat(
        [v["exploded"] for v in prepared_datasets.values()],
        ignore_index=True
    )
    combined_errors = pd.concat(
        [v["errors"] for v in prepared_datasets.values()],
        ignore_index=True
    )
    info = {
        "has_classification": any(v["info"].get("has_classification", False) for v in prepared_datasets.values()),
        "file_type": "combined",
        "file_count": sum(v["info"].get("file_count", 0) for v in prepared_datasets.values())
    }
    return {"analysis": combined_analysis, "exploded": combined_exploded, "errors": combined_errors, "info": info}

def generate_summary_stats(prepared_datasets):
    """
    Build a summary DataFrame with one row per dataset.
    """
    rows = []
    for name, data in prepared_datasets.items():
        info = data["info"]
        rows.append({
            "Dataset": name,
            "Has Classification": "Yes" if info.get("has_classification") else "No",
            "File Type": info.get("file_type"),
            "Responses": len(data["analysis"]),
            "Exploded": len(data["exploded"]),
            "Errors": len(data["errors"])
        })
    return pd.DataFrame(rows)