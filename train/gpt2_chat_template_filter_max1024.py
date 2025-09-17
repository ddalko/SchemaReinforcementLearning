import fire
import json
import re
from typing import Any, Set


from tqdm import tqdm
from transformers import AutoTokenizer

from filter_utils import project_schema_to_gt

tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.eos_token is None or tokenizer.eos_token != "<|endoftext|>":
    tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
if tokenizer.pad_token is None or tokenizer.pad_token != "[PAD]":
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
tokenizer.padding_side = "right"

ROLE_PREFIX = {
    "system": "### System:\n",
    "user": "### User:\n",
    "assistant": "### Assistant:\n",
}
EOS = tokenizer.encode(tokenizer.eos_token)[0]
CODEBLOCK_SCHEMA_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

def extract_schema_json(text: str) -> dict:
    """
    ```json ... ``` 코드블록 안의 첫 번째 JSON을 스키마로 파싱
    """
    m = CODEBLOCK_SCHEMA_RE.search(text)
    schema_str = m.group(1).strip()
    try:
        return json.loads(schema_str)
    except json.JSONDecodeError as e:
        print(f"코드블록 내 JSON 파싱 실패: {e}\n")
        return None

def replace_first_json_codeblock(text: str, obj: dict) -> str:
    new_json_str = json.dumps(obj, ensure_ascii=False, separators=(',', ':'))
    pattern = re.compile(r"(```json\s*)(.*?)(\s*```)", re.IGNORECASE | re.DOTALL)
    def repl(m):
        return f"{m.group(1)}{new_json_str}{m.group(3)}"
    return pattern.sub(repl, text, count=1)

def remove_keys_recursive(obj: Any, remove_keys: Set[str] = {"$schema", "$ref", "$id", "required"}) -> Any:
    """obj 내 모든 dict에서 remove_keys를 제거한 '새 객체'를 반환."""
    if isinstance(obj, dict):
        return {
            k: remove_keys_recursive(v, remove_keys)
            for k, v in obj.items()
            if k not in remove_keys
        }
    elif isinstance(obj, list):
        return [remove_keys_recursive(x, remove_keys) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(remove_keys_recursive(x, remove_keys) for x in obj)
    # str / int / float / bool / None 등은 그대로
    return obj


def render_messages_for_gpt2(messages):
    """
    메시지 배열을 GPT-2 친화적 텍스트로 변환.
    반환:
      text: 최종 합친 문자열
      segments: [(start_char, end_char, role), ...] 각 segment의 문자 인덱스 구간과 role
    """
    # BOS/EOS는 토큰 단계에서 처리. 여기선 순수 텍스트만 만든다.
    parts = []
    segments = []
    cursor = 0

    for m in messages:
        role = m["role"].lower()
        content = m.get("content", "")
        prefix = ROLE_PREFIX.get(role, f"### {role.capitalize()}:\n")
        if role == "assistant":
            content = content.replace(' ', '').replace('\n', '') # TODO: 빈 칸은 1개로 줄이기
        seg_text = prefix + content.strip() + "\n\n"
        parts.append(seg_text)
        start = cursor
        cursor += len(seg_text)
        end = cursor
        segments.append((start, end, role))

    text = "".join(parts)
    return text, segments

def convert_valid_compatible(schema: dict) -> dict:
    """
    1. "text": chat template 으로 구성
    content = 'Please generate a valid json object according to the following schema:\n\n```json\n{schema["model_schema"]}```'
    messages = [{'role': 'user', 'content': content}]
    2. "question_type": ["COMPLEX", "CUSTOM", "ESCAPE"] 중 하나
    3. 나머지 데이터 그대로 살림.
    """
    chat_template_data = []
    for question_type, data in schema.items():
        for item in data:
            json_schema = item if question_type == 'COMPLEX' else item['model_schema']
            content = f'Please generate a valid json object according to the following schema:\n\n```json\n{json_schema}```'
            temp = {
                'messages': [{'role': 'user', 'content': content}],
                'question_type': question_type,
            }
            if question_type != 'COMPLEX':
                temp.update({
                    'verify_schema': item['verify_schema'],
                })
            chat_template_data.append(temp)
    return chat_template_data

def main(
    data_path: str,
    filtered_data_save_path: str,
    max_token_length: int = 1024,
):
    with open(data_path, "r") as jfd:
        data = json.load(jfd)

    preprocessed_data = []
    data = [item for item in data if '```json\n{\"$schema' in item['messages'][1]['content']]
    for i in tqdm(range(len(data)), desc="Preprocessing dataset"):
        messages = data[i]["messages"]
        text, segs = render_messages_for_gpt2(messages)
        text = re.sub(r'(?m)^### System:\s*(\r?\n)*', '', text, count=1)

        schema_json = extract_schema_json(text)
        if not schema_json:
            continue
        gt = json.loads(text[segs[2][0]:segs[2][1]])
        minimal = project_schema_to_gt(schema_json, gt)
        text = replace_first_json_codeblock(text, minimal)

        tokens = tokenizer(text, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
        tokens['input_ids'].append(EOS)

        if len(tokens['input_ids']) > max_token_length:
            continue

        assistant_prefix = '### Assistant:\n'
        cut_point = text.find(assistant_prefix) + len(assistant_prefix)
        prompt = text[:cut_point].rstrip()

        tmp = {"text": text, "prompt": prompt}
        preprocessed_data.append(tmp)

    print(f"Filtered {len(preprocessed_data)} samples from {len(data)}")
    with open(filtered_data_save_path, "w", encoding="utf-8") as f:
        json.dump(preprocessed_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    fire.Fire(main)