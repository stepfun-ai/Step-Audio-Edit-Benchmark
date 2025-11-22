import json
import argparse
from tqdm import tqdm
import pandas as pd
import os

""" 
jsonl demo
{
    "id": "[Surprise-oh]-502292-301", 
    "speaker": "502292", 
    "gen_text": "You baked a cake...", 
    "prompt_audio": "prompt_audios/502292.wav", 
    "prompt_text": "he also engaged himself...", 
    "lang": "en", 
    "task": "paralingustic", 
    "task_sub": "[Surprise-oh]",
    "audio_path": "[Surprise-oh]-502292-301.wav",
    "gemini_score": "3"
}

use demo
    python3 get_gemini_paralingustic_score.py --gemini_res_jsonl paralingustic_res.jsonl --output_excel paralingustic_metric.xlsx
"""

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemini_res_jsonl", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output_excel", type=str, default=None, help="Output Excel file path")
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    print(f"Reading file: {args.gemini_res_jsonl}")
    
    if not os.path.exists(args.gemini_res_jsonl):
        print(f"Error: file not exist {args.gemini_res_jsonl}")
        return

    with open(args.gemini_res_jsonl, 'r', encoding='utf-8') as f:
        gemini_res_lines = f.readlines()
    
    res = {} 
    
    for line in tqdm(gemini_res_lines):
        line = line.strip()
        if not line:
            continue
            
        try:
            sample = json.loads(line)
        except json.JSONDecodeError:
            continue

        if 'gemini_score' not in sample or 'task_sub' not in sample:
            continue

        task_sub = sample['task_sub'] # eg "[Surprise-oh]"
        speaker = sample.get('speaker', 'unknown')
        
        try:
            score = float(sample['gemini_score'])
        except (ValueError, TypeError):
            continue

        if speaker not in res:
            res[speaker] = {}
        if task_sub not in res[speaker]:
            res[speaker][task_sub] = []
        
        res[speaker][task_sub].append(score)
    
    speaker_list = sorted(list(res.keys()))
    
    task_sub_set = set()
    for spk in res:
        for tag in res[spk]:
            task_sub_set.add(tag)
    task_sub_list = sorted(list(task_sub_set))
    
    if args.output_excel:
        table_data = []
        for speaker in speaker_list:
            row = {'speaker': speaker}
            
            speaker_all_scores = [] 
            
            for task_sub in task_sub_list:
                if task_sub in res[speaker]:
                    scores = res[speaker][task_sub]
                    avg_score = sum(scores) / len(scores)
                    row[task_sub] = round(avg_score, 2)
                    
                    speaker_all_scores.extend(scores)
                else:
                    row[task_sub] = None 

            if speaker_all_scores:
                row['OVERALL_AVG'] = round(sum(speaker_all_scores) / len(speaker_all_scores), 4)
            else:
                row['OVERALL_AVG'] = None
                
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        cols = ['speaker', 'OVERALL_AVG'] + [c for c in df.columns if c not in ['speaker', 'OVERALL_AVG']]
        df = df[cols]

        print(f"\nSaving Excel to: {args.output_excel}")
        df.to_excel(args.output_excel, index=False)
        print("Successfully saved")

        total_avg = df['OVERALL_AVG'].mean()
        print(f"Average Scores: {total_avg:.4f}")

if __name__ == "__main__":
    main()
