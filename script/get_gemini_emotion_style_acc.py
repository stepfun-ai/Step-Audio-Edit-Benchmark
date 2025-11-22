import json
import argparse
from tqdm import tqdm
import pandas as pd

# emotion_list = ["happy", "angry", "sad", "surprised", "fear"]
# style_list = ['child', 'exaggerated', 'recite', 'generous', 'act_coy', 'older', 'whisper']

""" 
jsonl demo
{
    "id": "surprised-20870-30", 
    "speaker": "20870", 
    "gen_text": "Only 29.95 dollars? I'll buy it, of course.", 
    "prompt_audio": "prompt_audios/20870.wav", 
    "prompt_text": "the committee has an independent secretariat which was previously provided by the cabinet office.", 
    "lang": "en", 
    "task": "emotion", 
    "task_sub": "surprised",
    "audio_path": "iter_3/surprised-20870-30.wav",
    "gemini_res": "surprised",
}

use demo
    python3 get_gemini_emotion_style_acc.py --gemini_res_jsonl dataset_gemini.jsonl --iters "0,1,2,3" --output_excel dataset_gemini.xlsx
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemini_res_jsonl", type=str, required=True)
    parser.add_argument("--iters", type=str, default="0,1,2,3")
    parser.add_argument("--output_excel", type=str, default=None)
    args = parser.parse_args()
    return args

def get_round_from_path(audio_path, iter_list):
    for iter in iter_list:
        if f"iter_{iter}" in audio_path:
            return iter

def main():
    args = get_args()
    iter_list = args.iters.split(",")

    with open(args.gemini_res_jsonl, 'r', encoding='utf-8')as f:
        gemini_res_lines = f.readlines()
    
    res = {iter: {} for iter in iter_list} 
    for line in tqdm(gemini_res_lines):
        sample = json.loads(line.strip())
        if 'gemini_res' not in sample:
            continue

        task_sub = sample['task_sub']
        gemini_res = sample['gemini_res']
        speaker = sample['speaker']
        audio_path = sample['audio_path']
        iter = get_round_from_path(audio_path, iter_list)

        if speaker not in res[iter]:
            res[iter][speaker] = {}
        if task_sub not in res[iter][speaker]:
            res[iter][speaker][task_sub] = []
        if gemini_res == task_sub or (gemini_res=="act_cute" and task_sub=="act_coy") or (gemini_res=="whisper (ASMR)" and task_sub=="whisper"):    
            res[iter][speaker][task_sub].append(1)
        else:
            res[iter][speaker][task_sub].append(0)
    
    speaker_set = set()
    task_sub_set = set()
    for iter in res:
        for speaker in res[iter]:
            speaker_set.add(speaker)
            for task_sub in res[iter][speaker]:
                task_sub_set.add(task_sub)
                per_task_sub_cnt = len(res[iter][speaker][task_sub])
                gemini_currect_cnt = sum(res[iter][speaker][task_sub])
                print(f"[iter-{iter}] {speaker} {task_sub} {gemini_currect_cnt/per_task_sub_cnt}")
    speaker_list = sorted(list(speaker_set))
    task_sub_list = sorted(list(task_sub_set))
    
    if args.output_excel:
        table_data = []
        for speaker in speaker_list:
            row = {'speaker': speaker}
            for task_sub in task_sub_list:
                for iter in iter_list:
                    if iter not in res:
                        row[f'{task_sub}_{iter}'] = 0
                    else:
                        row[f'{task_sub}_{iter}'] = sum(res[iter][speaker][task_sub]) / len(res[iter][speaker][task_sub]) * 100
            table_data.append(row)
            for iter in iter_list:
                total_score = 0.0
                for task_sub in task_sub_list:
                    total_score += row[f'{task_sub}_{iter}']
                row[f'iter_{iter}'] = round(total_score / len(task_sub_list), 4)
        
        df = pd.DataFrame(table_data)
        df.to_excel(args.output_excel, index=False)
        return df
        
if __name__ == "__main__":
    main()
