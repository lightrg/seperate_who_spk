from __future__ import annotations

import argparse
import logging
import unicodedata
from pathlib import Path

import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    set_seed,
)

from config import DEFAULT_LANGUAGE, DEFAULT_MODEL_NAME, DEFAULT_TASK, STAGE1_SPEC, STAGE2_SPEC
from dataloader import WhisperDataCollator, WhisperManifestDataset
from manifest_builder import export_pipeline_manifests, scan_all_sources

LOGGER = logging.getLogger("finetune")

def build_model_and_processor(model_name: str, language: str, task: str, lora_r: int, lora_alpha: int, lora_dropout: float):
    processor = WhisperProcessor.from_pretrained(model_name)
    processor.tokenizer.set_prefix_tokens(language=language, task=task)

    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    model.config.use_cache = False

    for param in model.parameters():
        param.requires_grad = False

    num_unfrozen = 0
    all_param = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        all_param += param.numel()
        if "q_proj" in name or "v_proj" in name:
            param.requires_grad = True
            trainable_params += param.numel()
            num_unfrozen += 1

    LOGGER.info(f"Unfrozen {num_unfrozen} parameter tensors (q_proj, v_proj) for partial finetuning.")
    LOGGER.info(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}")

    return model, processor

def normalize_vi_text(text: str) -> str:
\
\
\
\
\
\
\

    text = unicodedata.normalize("NFC", text)
    text = text.lower()

    text = "".join(ch if (ch.isalpha() or ch.isdigit() or ch.isspace()) else " " for ch in text)
    return " ".join(text.split())

def edit_distance(r, h):
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1): d[i][0] = i
    for j in range(len(h) + 1): d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(r)][len(h)]

def compute_wer(reference, hypothesis):
    total_edits = 0
    total_words = 0
    for r, h in zip(reference, hypothesis):
        r_words = r.split()
        h_words = h.split()
        total_edits += edit_distance(r_words, h_words)
        total_words += len(r_words)
    if total_words == 0:
        return 0.0
    return total_edits / total_words

def build_compute_metrics(processor: WhisperProcessor):
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = [normalize_vi_text(x) for x in processor.batch_decode(pred_ids, skip_special_tokens=True)]
        label_str = [normalize_vi_text(x) for x in processor.batch_decode(label_ids, skip_special_tokens=True)]
        wer_val = 100.0 * compute_wer(label_str, pred_str)
        return {"wer": wer_val}

    return compute_metrics

def train_one_stage(
    stage_name: str,
    stage_epochs: int,
    model_name_or_path: str,
    train_manifest: str,
    val_manifest: str,
    output_dir: str,
    args: argparse.Namespace,
) -> Path:
    model, processor = build_model_and_processor(
        model_name=model_name_or_path,
        language=args.language,
        task=DEFAULT_TASK,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    train_ds = WhisperManifestDataset(train_manifest, processor)
    val_ds = WhisperManifestDataset(val_manifest, processor)
    collator = WhisperDataCollator(processor)

    stage_out = Path(output_dir) / stage_name
    stage_out.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(stage_out),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=stage_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        fp16=args.fp16 and torch.cuda.is_available(),
        bf16=args.bf16 and torch.cuda.is_available(),
        dataloader_num_workers=args.num_workers,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        remove_unused_columns=False,
        report_to="none",
        label_names=["labels"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=processor.tokenizer,
        compute_metrics=build_compute_metrics(processor),
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    best_dir = stage_out / "best_adapter"
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))
    return best_dir

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PhoWhisper-large LoRA fine-tuning pipeline")
    parser.add_argument("--self_labeled_root", type=str, required=True)
    parser.add_argument("--dubbed_root", type=str, default="")
    parser.add_argument("--vivos_root", type=str, default="")
    parser.add_argument("--robustness_root", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--language", type=str, default=DEFAULT_LANGUAGE)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--generation_max_length", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--stage2_lr_scale", type=float, default=0.2,
                        help="Multiply --learning_rate by this factor for stage 2 to avoid catastrophic forgetting "
                             "(default 0.2 → 5× smaller than stage 1).")

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default="")
    parser.add_argument("--skip_stage2", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    set_seed(args.seed)

    LOGGER.info("Step 1/5: scan raw sources")
    all_data = scan_all_sources(
        self_labeled_root=args.self_labeled_root,
        dubbed_root=args.dubbed_root,
        vivos_root=args.vivos_root,
        robustness_root=args.robustness_root,
    )

    LOGGER.info("Step 2/5: build unified manifest, dedup, control repetition, export stage manifests")
    manifest_paths = export_pipeline_manifests(
        output_dir=args.output_dir,
        all_data=all_data,
        stage1_spec=STAGE1_SPEC,
        stage2_spec=STAGE2_SPEC,
        seed=args.seed,
    )

    LOGGER.info("Step 3/5: stage1 fine-tune")
    stage1_best = train_one_stage(
        stage_name=STAGE1_SPEC.name,
        stage_epochs=STAGE1_SPEC.epochs,
        model_name_or_path=args.model_name,
        train_manifest=str(manifest_paths["stage1_train"]),
        val_manifest=str(manifest_paths["stage1_val"]),
        output_dir=args.output_dir,
        args=args,
    )

    if args.skip_stage2:
        LOGGER.info("Skip stage2 requested. Best checkpoint: %s", stage1_best)
        return

    LOGGER.info("Step 4/5: stage2 robustness fine-tune")

    original_lr = args.learning_rate
    args.learning_rate = original_lr * args.stage2_lr_scale
    LOGGER.info("Stage 2 learning rate: %.2e (stage 1 lr=%.2e × scale=%.2f)",
                args.learning_rate, original_lr, args.stage2_lr_scale)
    stage2_best = train_one_stage(
        stage_name=STAGE2_SPEC.name,
        stage_epochs=STAGE2_SPEC.epochs,
        model_name_or_path=str(stage1_best),
        train_manifest=str(manifest_paths["stage2_train"]),
        val_manifest=str(manifest_paths["stage2_val"]),
        output_dir=args.output_dir,
        args=args,
    )
    args.learning_rate = original_lr

    LOGGER.info("Step 5/5: checkpoint selected by validation WER. Final best checkpoint: %s", stage2_best)

if __name__ == "__main__":
    main()