import requests 
import json
def process_api_requests_from_file(
    requests_filepath,
    save_filepath,
    request_url,
    api_key,
    token_encoding_name,
    max_attempts,
    proxy=None,
):
    with open(requests_filepath, 'r', encoding='utf-8') as file:
        for line in file:
            content = json.loads(line)
            task_id, target, target_index, request_json = content["task_id"], content["target"], content["target_index"], content["request"]
            pos = content.get("pos", None)
            prompt=request_json["messages"][0]["content"]
            content_json = {"contents":[{"parts":[{"text":prompt}]}],"generationConfig":{"temperature":0.0}}
            attempts_left = max_attempts
            while attempts_left > 0:
                try:
                    response = requests.post(
                        url=request_url,
                        headers={"Content-Type": "application/json"},
                        json=content_json,
                        proxies={"http": proxy, "https": proxy} if proxy else None
                    )

                    response.raise_for_status()  
                    append_to_jsonl({
                        "task_id": task_id,
                        "target": target,
                        "target_index": target_index,
                        "pos": pos,
                        "request": request_json,
                        "response": response.json()['candidates'][0]['content']['parts'][0]['text']
                    }, save_filepath)
                    break  
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:  
                        time.sleep(15)  
                    else:
                        print(f"HTTPError: {e}")
     
                        break  
                except Exception as e:
                    print(f"Exception: {e}")
                    break  
                finally:
                    attempts_left -= 1
def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")
