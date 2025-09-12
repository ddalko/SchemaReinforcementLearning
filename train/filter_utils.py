#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, re, sys
import copy
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, List

# ---------------------------
# Util: codeblock & assistant JSON 추출 (raw 텍스트 모드용)
# ---------------------------

FENCE_RE = re.compile(r"```json\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
BRACKET_RE = re.compile(r"\{|\}")

def safe_clone(obj):
    """
    JSON-호환(dict/list/str/num/None) 구조를 안전하게 복제.
    - 1차: copy.deepcopy 시도
    - 실패: json round-trip (사이클/비직렬화 객체 차단)
    - 그래도 실패: 얕은 복제 (마지막 안전망)
    """
    try:
        return copy.deepcopy(obj)
    except Exception:
        try:
            return json.loads(json.dumps(obj))
        except Exception:
            try:
                return copy.copy(obj)
            except Exception:
                return obj  # 최후의 수단: 원본 그대로 (변형 금지 전제)

def extract_first_codeblock_json(text: str) -> Optional[dict]:
    m = FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Fallback: Codeblock [[json ...]]
    m2 = re.search(r"Codeblock\s*\[\[\s*json\s*(.*?)\s*\]\]", text, re.IGNORECASE | re.DOTALL)
    if m2:
        try:
            return json.loads(m2.group(1))
        except Exception:
            pass
    return None

def extract_first_assistant_json(text: str) -> Optional[dict]:
    # Find "Assistant:" 이후 첫 JSON object (중괄호 밸런싱)
    idx = re.search(r"(###\s*Assistant:|Assistant:)", text, re.IGNORECASE)
    start = idx.end() if idx else 0
    s = text[start:]
    # naive scan for first {...} balanced
    stack = 0; begin = -1
    for m in BRACKET_RE.finditer(s):
        ch = m.group(0)
        if ch == "{":
            if stack == 0:
                begin = m.start()
            stack += 1
        else:
            stack -= 1
            if stack == 0 and begin != -1:
                frag = s[begin:m.end()]
                try:
                    return json.loads(frag)
                except Exception:
                    begin = -1
    return None

# ---------------------------
# Schema projection core
# ---------------------------

REMOVE_KEYS_FINAL = {
    "$schema", "$ref", "$id", "required", "definitions", "$defs", "components"
}

def remove_keys_recursive(obj: Any, remove_keys=REMOVE_KEYS_FINAL) -> Any:
    if isinstance(obj, dict):
        return {k: remove_keys_recursive(v, remove_keys) for k, v in obj.items() if k not in remove_keys}
    if isinstance(obj, list):
        return [remove_keys_recursive(x, remove_keys) for x in obj]
    return obj

def json_pointer_get(root: Any, ref: str) -> Optional[Any]:
    if not (isinstance(ref, str) and ref.startswith("#")):
        return None
    ptr = ref[1:]
    if ptr.startswith("/"):
        ptr = ptr[1:]
    if ptr == "":
        return root
    cur = root
    for raw_tok in ptr.split("/"):
        tok = raw_tok.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, dict) and tok in cur:
            cur = cur[tok]
        elif isinstance(cur, list) and tok.isdigit():
            i = int(tok)
            if 0 <= i < len(cur):
                cur = cur[i]
            else:
                return None
        else:
            return None
    return cur

def resolve_ref_once(node: dict, full_schema: dict) -> dict:
    """$ref 1-step 해제 + 형제키 overlay. 안전 복제 사용."""
    if not (isinstance(node, dict) and "$ref" in node):
        return node
    ref = node["$ref"]
    target = json_pointer_get(full_schema, ref)
    if not isinstance(target, dict):
        return node
    repl = safe_clone(target)  # <-- deepcopy 대신 안전 복제
    # overlay: node 쪽 형제 키가 우선
    for k, v in list(node.items()):
        if k != "$ref":
            repl[k] = v
    return repl

def resolve_refs_deep(node, full_schema: dict, stack=None):
    """
    $ref를 재귀적으로 푸는데, 같은 ref 경로 재방문을 차단하여 순환 방지.
    - $ref가 비표준(dict/list 등)인 경우도 안전하게 처리(해시 가능한 key로 변환)
    """
    if stack is None:
        stack = set()

    def _refkey(ref):
        # 순환 감지용 해시 가능한 키로 변환
        if isinstance(ref, str):
            return ("str", ref)
        # dict/list/tuple 등 비해시형 -> JSON 직렬화해서 키로 사용, 실패 시 id fallback
        try:
            import json
            return ("json", json.dumps(ref, sort_keys=True, separators=(",", ":")))
        except Exception:
            return ("id", id(ref))

    if isinstance(node, dict):
        out = node
        if "$ref" in out:
            ref = out["$ref"]
            key = _refkey(ref)
            if key in stack:
                # 순환 감지 → 더 풀지 않고 그대로 반환
                return out
            stack.add(key)
            # ref가 비표준이면 resolve_ref_once가 그대로 반환하므로 안전
            out = resolve_ref_once(out, full_schema)

        # 자식들 처리 (복제본에만 적용)
        out = safe_clone(out)
        for k, v in list(out.items()):
            out[k] = resolve_refs_deep(v, full_schema, stack)
        return out

    elif isinstance(node, list):
        return [resolve_refs_deep(x, full_schema, stack) for x in node]

    return node

def merge_allOf(node: dict) -> dict:
    """allOf를 (가능한 한) 얕게 병합. properties/type/enum/…을 보수적으로 합침."""
    node = deepcopy(node)
    if not (isinstance(node, dict) and "allOf" in node and isinstance(node["allOf"], list)):
        return node
    merged: Dict[str, Any] = {}
    # 현 노드의 properties/type 등도 포함
    def shallow_merge(dst: dict, src: dict):
        for k, v in src.items():
            if k == "properties" and isinstance(v, dict):
                dst.setdefault("properties", {})
                dst["properties"].update(v)
            elif k in {"anyOf", "oneOf", "allOf"}:
                # 일단 두면 후속 로직이 처리
                dst[k] = v
            else:
                dst[k] = v

    shallow_merge(merged, {k: v for k, v in node.items() if k != "allOf"})
    for part in node["allOf"]:
        if isinstance(part, dict):
            shallow_merge(merged, part)
    # 한 번 정리
    if "allOf" in merged:
        del merged["allOf"]
    return merged

def score_property_schema(prop_schema: dict, gt_value: Any) -> int:
    """후보 점수: enum 매치>type 존재>format/description 등."""
    if not isinstance(prop_schema, dict): return -1
    s = 0
    t = prop_schema.get("type")
    if t is not None: s += 2
    if "description" in prop_schema: s += 1
    if "format" in prop_schema: s += 1
    if "enum" in prop_schema and isinstance(prop_schema["enum"], list):
        if gt_value is not None and gt_value in prop_schema["enum"]:
            s += 5
        else:
            s += 1
    # object/array 같은 구조적 정보 있으면 가산
    if "properties" in prop_schema: s += 2
    if "items" in prop_schema: s += 2
    return s

def find_property_schema_anywhere(schema_root: dict, key: str, gt_value):
    best = None
    best_score = -1

    visited_nodes = set()   # id(node) 중복 방문 방지
    visited_refs = set()    # 동일 ref 반복 해제 방지

    def _refkey(_ref):
        # 순환 감지용 해시 가능한 키로 변환
        if isinstance(_ref, str):
            return ("str", _ref)
        # dict/list/tuple 등 비해시형 -> JSON 직렬화해서 키로 사용, 실패 시 id fallback
        try:
            import json
            return ("json", json.dumps(_ref, sort_keys=True, separators=(",", ":")))
        except Exception:
            return ("id", id(_ref))

    def visit(node):
        nonlocal best, best_score

        # 노드 재방문 차단
        try:
            nid = id(node)
            if nid in visited_nodes:
                return
            visited_nodes.add(nid)
        except Exception:
            pass

        if isinstance(node, dict):
            # $ref 한 번만 안전하게 해제하되, 같은 ref 반복 방지
            n = node
            if "$ref" in n:
                ref = _refkey(n["$ref"])
                if ref in visited_refs:
                    # 이미 본 ref → 더 깊게 안 들어감
                    return
                visited_refs.add(ref)
                n = resolve_ref_once(n, schema_root)

            n = merge_allOf(n)  # 얕은 병합 (여기서 deepcopy 안 씀)

            # 현 레벨에서 properties[key]
            props = n.get("properties")
            if isinstance(props, dict) and key in props:
                cand = resolve_refs_deep(props[key], schema_root)
                cand = merge_allOf(cand) if isinstance(cand, dict) else cand
                sc = score_property_schema(cand, gt_value)
                if sc > best_score:
                    best, best_score = cand, sc

            # 조합형 분기
            for comb in ("oneOf", "anyOf"):
                if comb in n and isinstance(n[comb], list):
                    for opt in n[comb]:
                        visit(opt)

            # 일부 거대 섹션은 스킵(무한/폭발 방지)
            for k, v in n.items():
                if k in ("definitions", "$defs", "components"):
                    continue
                visit(v)

        elif isinstance(node, list):
            for x in node:
                visit(x)

    visit(schema_root)
    return safe_clone(best) if best is not None else None


def guess_minimal_from_value(value: Any) -> dict:
    if isinstance(value, bool): return {"type": "boolean"}
    if isinstance(value, int): return {"type": "integer"}
    if isinstance(value, float): return {"type": "number"}
    if isinstance(value, str): return {"type": "string"}
    if isinstance(value, list):
        item_schema = guess_minimal_from_value(value[0]) if value else {}
        return {"type": "array", "items": item_schema}
    if isinstance(value, dict):
        return {"type": "object", "properties": {}, "additionalProperties": False}
    return {}

def project_schema_to_gt(schema_in_block: dict, gt_json: Any) -> dict:
    """
    핵심: GT에 있는 key들만 남기는 새 '미니멀 스키마'를 생성.
    - 각 key의 프로퍼티는 스키마 트리 '어디서든' 찾아와($ref/oneOf/anyOf/allOf 얕게 처리),
      실제 정의(type/format/description/enum/… )만 추림
    - nested dict/list도 재귀적으로 projection
    """
    root = deepcopy(schema_in_block)

    def project(node_root: dict, gt: Any) -> dict:
        # GT dict → object 스키마 구성
        if isinstance(gt, dict):
            props: Dict[str, Any] = {}
            for k, v in gt.items():
                found = find_property_schema_anywhere(node_root, k, v)
                if found is None:
                    # 못 찾으면 값에서 최소 추정
                    found = guess_minimal_from_value(v)
                # nested 재귀 (object/array일 때)
                if isinstance(v, dict) and isinstance(found, dict):
                    found = deepcopy(found)
                    # properties가 있으면 그 안에서 또 projection
                    if isinstance(found.get("properties"), dict):
                        found = project(found, v)
                elif isinstance(v, list) and isinstance(found, dict) and found.get("type") == "array":
                    items = found.get("items")
                    # 대표 원소 하나로 projection
                    rep = next((el for el in v if el is not None), None)
                    if isinstance(items, dict) and rep is not None:
                        found["items"] = project(items, rep)
                props[k] = remove_keys_recursive(found)
            return {"type": "object", "properties": props, "additionalProperties": False}

        # GT list → array 스키마
        if isinstance(gt, list):
            # 스키마 트리에서 'items'를 어디선가 찾기 어려우면 값에서 추정
            item_schema = {}
            rep = next((el for el in gt if el is not None), None)
            if rep is not None:
                item_schema = project(schema_in_block, rep)
            return {"type": "array", "items": remove_keys_recursive(item_schema)}

        # 원자값 → 타입 추정
        return remove_keys_recursive(guess_minimal_from_value(gt))

    minimal = project(root, gt_json)
    return minimal
