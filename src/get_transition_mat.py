import os
import json

def get_transition_mat(folder_path, num_options, mode="alphabet"):
    def generate_frequencies_and_mapping(num_options, mode):
        if mode == 'alphabet':
            characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            if num_options > len(characters):
                raise ValueError(f"Number of options exceeds the available characters in {mode} mode. Maximum supported options are {len(characters)}")
            frequencies = {characters[i]: 0 for i in range(num_options)}
            posIdx_mapping = {i: characters[i] for i in range(num_options)}
        elif mode == 'numeric':
            frequencies = {str(i): 0 for i in range(1, num_options+1)}
            posIdx_mapping = {i: str(i+1) for i in range(num_options)}
        else:
            raise ValueError(f"Invalid mode {mode}. Choose from 'alphabet', 'numeric'.")
        return frequencies, posIdx_mapping

    def process_file(file_path, frequencies):
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    data = json.loads(line)

                    rank_order = data['response']['gpt_data']['choices'][0]['message']['content']
                    target_index=data["target_index"]
                    truth_position=data["original_positions_after_shuffle"].index(target_index)

                    if 'rank_order' in rank_order:
                        rank_data = json.loads(rank_order)
                        order = rank_data['rank_order']
                        first_letter = order[0]
                        if first_letter in frequencies.keys():
                            frequencies[first_letter] += 1
                except:
                    continue
        return frequencies
    def process_shuffled_file(file_path, frequencies):
        for key in frequencies.keys():
            frequencies[key]={key: 0 for key in frequencies.keys()}
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    data = json.loads(line)

                    rank_order = data['response']['gpt_data']['choices'][0]['message']['content']
                    target_index=data["target_index"]
                    truth_position=data["original_positions_after_shuffle"].index(target_index)

                    if 'rank_order' in rank_order:
                        rank_data = json.loads(rank_order)
                        order = rank_data['rank_order']
                        first_letter = order[0]
                        if first_letter in frequencies.keys():
                            frequencies[chr(ord('A')+truth_position)][first_letter] += 1
                except:
                    continue
        return frequencies

    frequencies, posIdx_mapping = generate_frequencies_and_mapping(num_options, mode)
    counts = {}
    all_keys=frequencies.keys()
    for key in all_keys:
        counts[key]={key: 0 for key in all_keys}

    for root, dirs, files in os.walk(folder_path):
        for dir in dirs:
            if 'posIdx@' in dir:
                posIdx = int(dir.split('_')[1].split('@')[1])
                file_path = os.path.join(root, dir,'response.jsonl')
                if posIdx!=-1:
                    fre = process_file(file_path, {key: 0 for key in frequencies.keys()})
                    counts[posIdx_mapping[posIdx]]=fre
                else:
                    fre = process_shuffled_file(file_path, {key: 0 for key in frequencies.keys()})
                    for truth_key,val in fre.items():
                        for k,v in val.items():
                            counts[truth_key][k] += v

    counts_sorted = dict(sorted(counts.items())) 
    return counts_sorted


