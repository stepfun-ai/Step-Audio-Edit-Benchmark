import base64
import json
import time
import os
import sys
import requests
import threading
import traceback
import re
import concurrent.futures
import argparse
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Gemini Audio Evaluation Script")
    parser.add_argument("input_jsonl", type=str, help="Input JSONL file path")
    parser.add_argument("task_type", type=str, help="Task type (e.g., paralinguistic, emotion, style)")
    parser.add_argument("api_key", type=str, help="Gemini API Key")
    parser.add_argument("--prompt_file", type=str, default="gemini_prompt.json", help="Path to the prompt config file")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of threads for concurrent processing")
    args = parser.parse_args()
    return args


def create_paralinguistic_prompt(generated_text, prompt_template, tag_map):
    """
    Args:
        generated_text (str): input text
        prompt_template (str): prompt template in json
        tag_map (dict): tags map in json
    """
    if not tag_map:
        print(f"[Error] Tag map is empty, please check the prompt file.")
        return None

    checkpoints = []
    processed_text = generated_text.strip()
    
    for m in re.finditer(r'\[.*?\]', processed_text):
        tag = m.group(0)
        start_index = m.start()
        
        tag_display = tag_map.get(tag, tag) 

        if start_index < 1:
            special_context = (f"**在音频最开始处**，请立刻、主动地搜索 **{tag_display}**。")
            checkpoints.append(special_context)
        else:
            context_start = max(0, start_index - 5)
            preceding_context = processed_text[context_start:start_index].replace('\n', ' ').strip()
            normal_context = (f"在 “...{preceding_context}” 这句话之后，请立刻、主动地搜索 **{tag_display}**。")
            checkpoints.append(normal_context)

    if not checkpoints:
        checklist_for_prompt = "此音频没有特定的副语言标签需要检查。"
    else:
        checklist_lines = []
        for i, check_instruction in enumerate(checkpoints):
            checklist_lines.append(f"  **考点 {i+1}:** {check_instruction}")
        checklist_for_prompt = "\n".join(checklist_lines)
    
    translated_generated_text = generated_text
    for original_tag, translated_tag in tag_map.items():
        translated_generated_text = translated_generated_text.replace(original_tag, translated_tag)

    try:
        formatted_prompt = prompt_template.format(
            translated_generated_text=translated_generated_text,
            checklist_for_prompt=checklist_for_prompt
        )
        return formatted_prompt
    except Exception as e:
        print(f"[Error] Failed to format prompt template: {e}")
        return None

def load_gemini_prompt(prompt_file):
    if not os.path.exists(prompt_file):
        return {}
    with open(prompt_file, 'r', encoding='utf-8') as inf:
        return json.load(inf)


def call_gemini_api_text(prompt, audio_path, task_type, max_retries=6, retry_delay=3):
    global current_api_url
    
    for attempt in range(max_retries):
        try:
            with open(audio_path, 'rb') as audio_file:
                audio_data = audio_file.read()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "audio/wav", "data": audio_base64}}
                    ]
                }]
            }

            if task_type != "paralinguistic":
                payload["generationConfig"] = {"response_mime_type": "application/json"}

            response = requests.post(current_api_url, json=payload, headers=headers, timeout=120)

            if response.status_code == 200:
                try:
                    res_json = json.loads(response.text)
                    content = res_json['candidates'][0]['content']['parts'][0]['text']
                    clean_text = content.replace("```json", "").replace("```", "").strip()
                    return clean_text
                except Exception:
                    pass
            
            print(f"{attempt + 1} times request failed: {response.status_code} {response.text[:200]}")
            if attempt < max_retries - 1: time.sleep(retry_delay)

        except Exception as e:
            print(f"{attempt + 1} times exception: {e}")
            traceback.print_exc()
            if attempt < max_retries - 1: time.sleep(retry_delay)
            
    return None


def process_one_line(line, done_set, task_type, static_config=None):
    try:
        sample = json.loads(line.strip())
        audio_path = sample['audio_path']
    except:
        return None

    if audio_path in done_set:
        return None
    
    if not os.path.exists(audio_path):
        print(f"[Warning] File not found: {audio_path}")
        return None

    if not static_config:
        return None

    # 1. 准备 Prompt
    prompt = ""
    if task_type == "paralinguistic":
        gen_text = sample.get('gen_text', '')
        template = static_config.get("prompt_template", "")
        tag_map = static_config.get("tag_map", {}) 
        if template:
            prompt = create_paralinguistic_prompt(gen_text, template, tag_map)
            print(f"[Info] Prompt: {prompt}")
    else:
        prompt = static_config.get("prompt", "")

    if not prompt:
        return None

    res_text = call_gemini_api_text(prompt, audio_path, task_type)

    if res_text:
        if task_type == "paralinguistic":
            sample['gemini_score'] = res_text
        else:
            try:
                res_json_content = json.loads(res_text)
                labels_list = static_config.get("labels", [])
                
                target_key = task_type 
                val = res_json_content.get(target_key)
                
                if not val and len(res_json_content) > 0:
                    val = list(res_json_content.values())[0]

                if val in labels_list or val == "None": 
                    sample['gemini_res'] = val
                else:
                    return None
            except json.JSONDecodeError:
                return None

        return json.dumps(sample, ensure_ascii=False) + "\n"
    
    return None


if __name__ == "__main__":
    args = get_args()
    
    current_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key={args.api_key}"

    input_file = args.input_jsonl
    task_type = args.task_type
    output_file = f"{input_file.rsplit('.')[0]}_gemini.jsonl"
    
    write_lock = threading.Lock()
    
    all_prompts = load_gemini_prompt(args.prompt_file)
    static_config = None
    
    if task_type in all_prompts:
        static_config = all_prompts[task_type]
    else:
        print(f"Error: Task [{task_type}] not found in {args.prompt_file}")
        sys.exit(1)

    done_set = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as ef:
            for line in ef:
                try:
                    d = json.loads(line.strip())
                    done_set.add(d['audio_path'])
                except: pass

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Processing {len(lines)} lines...")
    print(f"Task: {task_type}")
    print(f"Threads: {args.num_workers}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_file = {
            executor.submit(process_one_line, line, done_set, task_type, static_config): line
            for line in lines
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(lines), desc=f"gemini"):
            try:
                res = future.result()
                if res:
                    with write_lock:
                        with open(output_file, 'a', encoding='utf-8') as f:
                            f.write(res)
                            f.flush()
            except Exception as e:
                print(f"[Error] {e}")
