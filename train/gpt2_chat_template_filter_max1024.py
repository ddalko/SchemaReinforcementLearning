import json
import re
from typing import Any, List, Set

import fire
from filter_utils import project_schema_to_gt
from tqdm import tqdm
from transformers import AutoTokenizer

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
    
def insert_boundary_whitespace(s: str, mask: List[bool]):
    """Insert a single space before and after each contiguous value span (mask==False).
    Inserted spaces are considered structure (True) in the returned mask.
    Returns (new_s, new_mask).
    """
    n = len(s)
    out_chars = []
    out_mask = []
    i = 0
    while i < n:
        if i < len(mask) and not mask[i]:
            # insert space before value if previous char is not whitespace
            if len(out_chars) == 0 or not out_chars[-1].isspace():
                out_chars.append(' ')
                out_mask.append(True)
            # copy contiguous False span
            j = i
            while j < n and j < len(mask) and not mask[j]:
                out_chars.append(s[j])
                out_mask.append(False)
                j += 1
            # insert space after value if next original char is not whitespace
            if j < n and not s[j].isspace():
                out_chars.append(' ')
                out_mask.append(True)
            i = j
            continue
        # copy structure char
        out_chars.append(s[i])
        out_mask.append(True)
        i += 1
    return ''.join(out_chars), out_mask


def generate_structure_char_mask(s: str):
    """
    JSON 텍스트 s의 각 문자에 대해:
      - value(문자열/숫자/bool/null) 구간은 False
      - 그 외(키, 구분자, 중괄호 등)는 True
    를 반환한다 (mask = [True]*n 으로 시작 → value에서 False로 전환).

    추가 규칙:
      - 콜론 뒤 값 처리
      - 배열이 '문자열/숫자/bool/null'로만 이루어진 리스트라면
        그 배열 전체(대괄호, 쉼표 포함)를 전부 False로 마스킹
      - 빈 배열 [] 및 빈 객체 {}는 각각 대괄호/중괄호를 False 처리
    """
    n = len(s)
    mask = [True] * n

    def set_false(a: int, b: int):
        a = max(0, a); b = min(n - 1, b)
        if a <= b:
            for i in range(a, b + 1):
                mask[i] = False

    def skip_ws(i: int) -> int:
        while i < n and s[i].isspace():
            i += 1
        return i

    def string_end(i: int) -> int:
        # i는 여는 따옴표('"') 위치
        j = i + 1
        while j < n:
            c = s[j]
            if c == "\\":
                j += 2          # 이스케이프 다음 글자 스킵
            elif c == '"':
                return j
            else:
                j += 1
        return n - 1             # 비정상 종료 시 끝까지

    number_regex = re.compile(r'-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+\-]?\d+)?')

    def number_end(i: int) -> int:
        m = number_regex.match(s, i)
        return m.end() - 1 if m else i

    def literal_match(i: int, lit: str) -> bool:
        return s.startswith(lit, i)

    def primitive_only_array_end(i_lbrack: int):
        """
        s[i_lbrack] == '[' 인 위치에서,
        배열이 (문자열/숫자/true/false/null) 원소만으로 구성되어 있으면
        닫는 ']'의 인덱스를 반환. 아니면 None.
        공백과 쉼표는 허용.
        """
        i = skip_ws(i_lbrack + 1)
        if i >= n:
            return None
        # 빈 배열 []
        if s[i] == ']':
            return i

        while i < n:
            i = skip_ws(i)
            if i >= n:
                return None
            c = s[i]
            # 문자열 원소
            if c == '"':
                e = string_end(i)
                i = e + 1
            # 숫자 원소
            elif c in "-0123456789":
                e = number_end(i)
                i = e + 1
            # true/false/null
            elif literal_match(i, "true"):
                i += 4
            elif literal_match(i, "false"):
                i += 5
            elif literal_match(i, "null"):
                i += 4
            else:
                # 객체/배열/기타 토큰 등장 → primitive-only 아님
                return None

            i = skip_ws(i)
            if i >= n:
                return None
            if s[i] == ",":
                i += 1
                continue
            if s[i] == "]":
                return i  # 성공
            # 다른 토큰 → 실패
            return None

    prev_sig = None
    i = 0
    in_string = False

    while i < n:
        c = s[i]

        # 문자열 경계 관리(마스킹은 콜론/배열 진입 규칙에서 처리)
        if in_string:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_string = False
                prev_sig = '"'
            i += 1
            continue

        if c == '"':
            in_string = True
            i += 1
            continue

        if c.isspace():
            i += 1
            continue

        # ---------- 콜론 뒤 값 ----------
        if c == ":":
            j = skip_ws(i + 1)
            if j >= n:
                i += 1; prev_sig = ":"; continue
            nxt = s[j]

            # 배열 값
            if nxt == "[":
                arr_end = primitive_only_array_end(j)
                if arr_end is not None:
                    # 규칙 3: 원시값 전용 리스트 → 배열 전체 마스킹
                    set_false(j, arr_end)
                    i = arr_end + 1
                    prev_sig = "]"
                    continue
                # 빈 배열이 아니거나 원시 전용이 아니면, 그대로 진행(내부에서 다른 규칙으로 처리)
            # 객체 값
            if nxt == "{":
                j2 = skip_ws(j + 1)
                if j2 < n and s[j2] == "}":
                    # {} → 둘 다 False
                    set_false(j, j)     # '{'
                    set_false(j2, j2)   # '}'
                    i = j2 + 1
                    prev_sig = "}"
                    continue
                # 비어있지 않으면 일반 로직으로

            # 문자열 값
            if nxt == '"':
                e = string_end(j)
                set_false(j, e)
                i = e + 1
                prev_sig = '"'
                continue

            # 숫자 값
            if nxt in "-0123456789":
                e = number_end(j)
                set_false(j, e)
                i = e + 1
                prev_sig = s[e] if e < n else None
                continue

            # true / false / null
            if literal_match(j, "true"):
                set_false(j, j + 3)
                i = j + 4
                prev_sig = "e"
                continue
            if literal_match(j, "false"):
                set_false(j, j + 4)
                i = j + 5
                prev_sig = "e"
                continue
            if literal_match(j, "null"):
                set_false(j, j + 3)
                i = j + 4
                prev_sig = "l"
                continue

            # 그 외(객체/배열 등) → 내부에서 계속 처리
            i += 1
            prev_sig = ":"
            continue

        # ---------- 배열 원소 시작(prev_sig가 '[' 또는 ',') ----------
        # (배열이 값으로 오지 않아도, 최상위/중첩 배열 모두 커버)
        if prev_sig in ("[", ","):
            # 배열이 원시 전용이면 전체 마스킹(여기서도 인식)
            if c == "[":
                arr_end = primitive_only_array_end(i)
                if arr_end is not None:
                    set_false(i, arr_end)
                    i = arr_end + 1
                    prev_sig = "]"
                    continue
            # 개별 원소가 원시 리터럴이면 그 리터럴만 마스킹
            if c == '"':
                e = string_end(i)
                set_false(i, e)
                i = e + 1
                prev_sig = '"'
                continue
            if c in "-0123456789":
                e = number_end(i)
                set_false(i, e)
                i = e + 1
                prev_sig = s[e] if e < n else None
                continue
            if literal_match(i, "true"):
                set_false(i, i + 3)
                i = i + 4
                prev_sig = "e"
                continue
            if literal_match(i, "false"):
                set_false(i, i + 4)
                i = i + 5
                prev_sig = "e"
                continue
            if literal_match(i, "null"):
                set_false(i, i + 3)
                i = i + 4
                prev_sig = "l"
                continue

        # 구조 문자/구분자 등 prev_sig 갱신
        if c in "{}[],":
            prev_sig = c
        else:
            prev_sig = c

        i += 1

    return mask

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


def render_messages_for_gpt2(messages, add_white_space_to_gt: bool = False):
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
            if add_white_space_to_gt:
                mask = generate_structure_char_mask(content)
                content, _ = insert_boundary_whitespace(content, mask)
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
    add_white_space_to_gt: bool = False,
    max_token_length: int = 1024,
):
    with open(data_path, "r") as jfd:
        data = json.load(jfd)

    preprocessed_data = []
    data = [item for item in data if '```json\n{\"$schema' in item['messages'][1]['content']]
    for i in tqdm(range(len(data)), desc="Preprocessing dataset"):
        messages = data[i]["messages"]
        text, segs = render_messages_for_gpt2(messages, add_white_space_to_gt)
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