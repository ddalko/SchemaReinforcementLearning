import json

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
        seg_text = prefix + content.strip() + "\n\n"
        parts.append(seg_text)
        start = cursor
        cursor += len(seg_text)
        end = cursor
        segments.append((start, end, role))

    text = "".join(parts)
    return text, segments

data_p = "train/data/mix_train_no_collected_json.json"
with open(data_p, "r") as jfd:
    data = json.load(jfd)

preprocessed_data = []
for i in tqdm(range(len(data)), desc="Preprocessing dataset"):
    messages = data[i]["messages"]
    if "```json\n{\"$schema" not in messages[1]['content']:
        continue
    text, segs = render_messages_for_gpt2(messages)

    tokens = tokenizer(text, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
    tokens['input_ids'].append(EOS)

    if len(tokens['input_ids']) > 1024:
        continue

    preprocessed_data.append({"text": text})

filtered_data_p = "train/data/chat_templated_jsonschema_dataset_max1024.json"
with open(filtered_data_p, "w", encoding="utf-8") as f:
    json.dump(preprocessed_data, f, ensure_ascii=False, indent=2)
