########################################################
# ANALYSIS HELPERS : TODO!
########################################################

def load_and_clean_single_run(csvfiles, Qs, add_run_label = None):
    single_df = pd.DataFrame()
    for csv_file in csvfiles:
        df = pd.read_csv(csv_file)
        df.drop("confidence", axis=1, inplace=True)
        single_df = pd.concat([single_df, df], ignore_index=True)

        single_df['answer_str'] = single_df['answer'].apply(str)
        single_df['answer'] = single_df['answer_str'].str.extract(r'(\d+)')
        single_df['answer'] = pd.to_numeric(single_df['answer'], errors='coerce')
    # add category to dataframe
    single_df['category'] = single_df['question_id'].apply(lambda x: Qs.get_question_category(str(x)))
    # add label
    if add_run_label:
            single_df['run_label'] = add_run_label
     
    return single_df


def get_model_shortname(model_name):
    result = re.split(r'[/_-]', model_name)
    return result[1] 

def ring_csv_to_df(csv_file, Qs):
    df = pd.read_csv(csv_file)
    df['category'] = df['question_id'].apply(lambda x: Qs.get_question_category(str(x)))
    json_columns = ['config_details', 'conversation_history', 'agent_responses']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    return df

def ring_to_roundrobin_df(ring_df, Qs): 
    # TODO: currently doesn't handle EVIL AGENTS 
    # ring_df should be for a single run (chat_type)
    if len(ring_df['chat_type'].unique()) != 1 :
        print('ring_df should have exactly 1 chat_type')
        return 
    

    # NOTE: MAY NEED TO CHANGE THIS IF THIS CHANGES
    # nubmer of rounds, models, and repeats assumes that all questions 
    # have the same rounds, the same models and the same number of repeats (and all models have the same number of repeats)
    n_rounds = ring_df['config_details'][0]['loops'] # number of rounds per round robin
    
    # n_models = len(ring_df['config_details'][0]['ensemble']) # number of models/agents (only works for hetero)
    n_models = sum([x['number'] for x in ring_df['config_details'][0]['ensemble']])

    # print(f'{n_models}')
    repeats = ring_df['run_index'].unique()
    
    # add 1 to repeat if starts at 0 else add 0 when saving
    minrep = min(repeats)
    if minrep == 0:
        add_to_repeat = 1
    elif minrep == 1:
        add_to_repeat = 0
    else:
        Warning(f'repeats start at {minrep}')
        add_to_repeat = 0

    n_repeats = repeats.max() # number of repeats (same question different round robin)

    if n_repeats == 1:
        round_robin_responses = ring_df['agent_responses']
    
    if n_repeats < 1:
        print('No repeats. Aborting')
        return

    # Build new dataframe for round robins
    rows = [] # list of dictionaries to go into df
    for repeat in repeats:
        if n_repeats > 1:
            round_robin_df = ring_df[(ring_df['run_index'] == repeat)]
            round_robin_responses = round_robin_df['agent_responses']
        
        # indices into round_robin dataframe (for question and repeat)
        q_indices = round_robin_df.index.to_list()    
        
        for idx in q_indices:
            q_num = round_robin_df.loc[idx]['question_num']
            q_id = round_robin_df.loc[idx]['question_id']
            
            for round in range(n_rounds):
                # get the message indices for this round
                message_indices = range(n_models*round + 1, n_models * (round+1)+1)
                # print(f'{message_indices}') # sanity check
                for msg_idx in message_indices:
                    # TODO for when nrepeats > 1 : 
                    # Add checks that the message ids and responses indices are correct 
                    dict_idx = msg_idx-1

                    # sanity check
                    # print(f'{repeat}, {idx},{dict_idx}')
                    message_idx = round_robin_responses[idx][dict_idx]['message_index']
                    agent_name = round_robin_responses[idx][dict_idx]['agent_name']
                    agent_model = round_robin_responses[idx][dict_idx]['agent_model']
                    agent_answer = round_robin_responses[idx][dict_idx]['extracted_answer']
                    agent_fullresponse = round_robin_responses[idx][dict_idx]['message_content']
                    rows.append({
                        'question_num' : q_num, 
                        'question_id': q_id, # starts at 1
                        'round': round+1, # starts at 1
                        'message_index': message_idx, # starts at 1
                        'agent_name': agent_name,
                        'agent_model':agent_model,
                        'agent_answer_str': agent_answer,
                        'agent_fullresponse': agent_fullresponse,
                        'repeat_index': repeat + add_to_repeat # should start at 1
                    })
        del round_robin_df
        del round_robin_responses

    ring_rr_df = pd.DataFrame(rows)

    # make sure are answers numeric
    ring_rr_df['agent_answer'] = ring_rr_df['agent_answer_str'].str.extract(r'(\d+)')
    ring_rr_df['agent_answer'] = pd.to_numeric(ring_rr_df['agent_answer'], errors='coerce')
    # add category to dataframe
    ring_rr_df['category'] = ring_rr_df['question_id'].apply(lambda x: Qs.get_question_category(str(x)))

    ring_rr_df['chat_type'] = ring_df['chat_type'].iloc[0]

    return ring_rr_df