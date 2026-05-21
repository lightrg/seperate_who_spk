
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Judge summary quality locally with Qwen2.5-7B-Instruct.")
    p.add_argument("--summary_json", required=True)
    p.add_argument("--transcript", required=True, help="Transcript jsonl or cleaned transcript text.")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--output_json", default=None)
    p.add_argument("--max_source_chars", type=int, default=22000)
    p.add_argument("--max_input_tokens", type=int, default=4096)
    p.add_argument("--max_new_tokens", type=int, default=900)
    p.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit loading if bitsandbytes is available.")
    p.add_argument("--device_map", default="auto")
    p.add_argument("--temperature", type=float, default=0.0)
    return p.parse_args()

def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_transcript_text(path: str | Path) -> str:
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        chunks: List[str] = []
        with open(p, "r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                obj = json.loads(raw)
                start = obj.get("start", obj.get("start_time", 0.0))
                end = obj.get("end", obj.get("end_time", start))
                speaker = obj.get("speaker", obj.get("speaker_id", "unknown"))
                text = obj.get("text", obj.get("sentence", ""))
                chunks.append(f"[{start}-{end}] {speaker}: {text}")
        return "\n".join(chunks)
    return Path(path).read_text(encoding="utf-8")

def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return head + "\n... [TRUNCATED] ...\n" + tail

def build_prompt(source_text: str, summary: Dict[str, Any]) -> List[Dict[str, str]]:
    condensed_summary = {
        "conversation_main_summary": summary.get("conversation_main_summary", ""),
        "meeting_overview": summary.get("meeting_overview", ""),
        "speaker_main_summaries": summary.get("speaker_main_summaries", []),
        "speaker_insights": summary.get("speaker_insights", []),
        "action_items": summary.get("action_items", []),
    }

    system = (
        "Bạn là giám khảo cực kỳ nghiêm khắc cho bài toán tóm tắt hội thoại nhiều người nói. "
        "Nhiệm vụ của bạn là chấm summary dựa trên transcript nguồn. "
        "Chỉ dùng bằng chứng có trong transcript. Không suy diễn vượt quá dữ liệu. "
        "Trả về JSON hợp lệ duy nhất, không có markdown, không có giải thích ngoài JSON."
    )

    user = f"""
Hãy chấm chất lượng summary sau trên thang điểm rõ ràng.

TRANSCRIPT NGUỒN:
{source_text}

SUMMARY CẦN CHẤM:
{json.dumps(condensed_summary, ensure_ascii=False, indent=2)}

Trả về JSON với schema này:
{json.dumps({
    "overall": {
        "main_summary_score": 0,
        "speaker_summary_score": 0,
        "faithfulness_score": 0,
        "conciseness_score": 0,
        "total_score": 0,
        "verdict": "pass|review|fail"
    },
    "issues": ["..."],
    "speaker_judgments": [
        {
            "speaker": "speaker_00",
            "score": 0,
            "supported": True,
            "reason": "..."
        }
    ],
    "final_comment": "..."
}, ensure_ascii=False, indent=2)}

Quy tắc chấm:
- main_summary_score (0-5): summary tổng có nêu đúng ý chính của cuộc hội thoại không.
- speaker_summary_score (0-5): phần tóm tắt theo speaker có đúng người, đúng nội dung chính không.
- faithfulness_score (0-5): có bịa, gộp nhầm, suy diễn quá mức không.
- conciseness_score (0-5): có ngắn gọn, đủ ý, không dài dòng không.
- total_score = tổng 4 điểm thành phần, tối đa 20.
- verdict:
  - pass nếu total_score >= 16
  - review nếu 10 <= total_score < 16
  - fail nếu total_score < 10

Lưu ý:
- Nếu bằng chứng không đủ, hãy nói rõ là không đủ bằng chứng.
- Nếu summary theo speaker bị lệch vai trò hoặc nhầm người, phải nêu rõ trong issues.
- Chỉ trả JSON.
""".strip()

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

def load_model_and_tokenizer(model_name: str, load_in_4bit: bool, device_map: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model_kwargs: Dict[str, Any] = {
        "device_map": device_map,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    if torch.cuda.is_available() or torch.backends.mps.is_available():
        model_kwargs["dtype"] = torch.float16
    if torch.backends.mps.is_available():
        model_kwargs["attn_implementation"] = "eager"

    if load_in_4bit and BitsAndBytesConfig is not None:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
        )
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    return tokenizer, model

def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        return json.loads(m.group(0))
    raise ValueError("Model output is not valid JSON")

def main() -> None:
    args = parse_args()
    summary = load_json(args.summary_json)
    transcript_text = load_transcript_text(args.transcript)
    source_text = truncate_text(transcript_text, args.max_source_chars)

    tokenizer, model = load_model_and_tokenizer(args.model_name, args.load_in_4bit, args.device_map)
    messages = build_prompt(source_text, summary)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_input_tokens,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if args.temperature > 0:
        gen_kwargs["temperature"] = args.temperature
    with torch.inference_mode():
        output = model.generate(**inputs, **gen_kwargs)
    new_tokens = output[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    judged = extract_json(text)

    judged_report = {
        "model_name": args.model_name,
        "summary_json": str(Path(args.summary_json).resolve()),
        "transcript": str(Path(args.transcript).resolve()),
        "judge_output": judged,
        "raw_text": text,
    }
    out_path = args.output_json or str(Path(args.summary_json).with_suffix(".qwen7b_judge.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(judged_report, f, ensure_ascii=False, indent=2)
    print(f"Saved judge report: {out_path}")

if __name__ == "__main__":
    main()
