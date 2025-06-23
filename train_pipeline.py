#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
from typing import Optional, Tuple

import torch
from accelerate import Accelerator
from flash_attn.losses.cross_entropy import CrossEntropyLoss
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import LlamaConfig, PreTrainedTokenizerFast

from model_llama import LlamaForCausalLM  # 本地 Flash-Attn 版 Llama

# ------------------------- 数据集 ------------------------- #


class AlpacaDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512):
        self.samples = [json.loads(l) for l in open(jsonl_path)]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prot = self.samples[idx]["input"]
        rna = self.samples[idx]["output"]
        text = f"{prot}<sep>{rna}"
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0)  # [seq_len]


def create_dataloaders(
    batch_size: int,
    seq_len: int,
    accelerator,
    train_path: str,
    val_path: Optional[str] = None,
    seed: int = 42,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    torch.manual_seed(seed)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        "/workspace/huangxiaoniu/diffu/aa_tokenizer"
    )
    tokenizer.add_tokens(["<sep>"])  # 追加分隔符
    train_ds = AlpacaDataset(train_path, tokenizer, seq_len)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)

    val_dl = None
    if val_path:
        val_ds = AlpacaDataset(val_path, tokenizer, seq_len)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_dl, val_dl, tokenizer


# ------------------------- Diffusion 掩码 ------------------------- #


def transition(x_0, sigma, maskable_mask, mask_token_id):
    """TinyLlama DDM token noising"""
    move_idx = (torch.rand_like(x_0.float()) < sigma) & maskable_mask
    return torch.where(move_idx, mask_token_id, x_0)


# ------------------------- 主程序 ------------------------- #


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulate_every,
        mixed_precision="bf16",
    )
    accelerator.print(f"Total GPUs: {accelerator.num_processes}")

    # --------- 数据 --------- #
    train_dl, _, tokenizer = create_dataloaders(
        args.batch_size,
        args.seq_length,
        accelerator,
        args.dataset,
    )

    # --------- 配置 & 模型 --------- #
    cfg = LlamaConfig.from_pretrained(args.model)
    cfg.rope_scaling = {"type": "none", "factor": 1.0}

    model = LlamaForCausalLM.from_pretrained(
        args.model,
        config=cfg,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
    )
    model.resize_token_embeddings(len(tokenizer))  # 为 <sep> 扩容

    # LoRA
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "embed_tokens",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.gradient_checkpointing_enable()

    # --------- Optim / Prepare --------- #
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = CrossEntropyLoss(inplace_backward=True, reduction="none")

    model, optim, train_dl = accelerator.prepare(model, optim, train_dl)

    # --------- 训练 --------- #
    sampling_eps = 1e-3
    mask_token_id = tokenizer.mask_token_id or tokenizer.eos_token_id
    accelerator.print(f"mask_token_id = {mask_token_id}")

    progress = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    step_done = 0
    model.train()

    for batch in train_dl:
        input_ids = batch.to(accelerator.device)[:, : args.seq_length]
        target_ids = input_ids.clone()
        pos_ids = (
            torch.arange(args.seq_length, device=input_ids.device)
            .unsqueeze(0)
            .expand_as(input_ids)
        )

        sigma = (1 - sampling_eps) * torch.rand(input_ids.size(0), device=input_ids.device) + sampling_eps
        dsigma = sigma.reciprocal()

        noised = transition(
            input_ids, sigma[:, None], maskable_mask=torch.ones_like(input_ids, dtype=torch.bool), mask_token_id=mask_token_id
        )
        loss_mask = noised.eq(mask_token_id)

        with accelerator.accumulate(model):
            logits = model(noised, position_ids=pos_ids).logits
            loss = loss_fn(
                logits.view(-1, logits.size(-1)), target_ids.view(-1)
            ).view_as(target_ids)
            loss = loss.masked_fill(~loss_mask, 0)
            loss = (dsigma[:, None] * loss).sum() / loss_mask.sum()

            accelerator.backward(loss)
            optim.step()
            optim.zero_grad()

            if accelerator.sync_gradients:
                loss_val = accelerator.gather(loss.detach()).mean().item()
                progress.update(1)
                progress.set_postfix(loss=f"{loss_val:.4f}", ppl=f"{math.exp(loss_val):.2f}")
                step_done += 1

        if step_done >= args.max_train_steps:
            break

    accelerator.end_training()
    accelerator.unwrap_model(model).save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    accelerator.print("Training finished & model saved.")


# ------------------------- CLI ------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulate-every", type=int, default=4)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-train-steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=5e-5)

    parser.add_argument("--model", type=str, default="/workspace/huangxiaoniu/diffu/DiffuLLaMA/llama-3.2-1B")
    parser.add_argument("--dataset", type=str, default="/workspace/huangxiaoniu/diffu/pipeline_data/train_data.jsonl")
    parser.add_argument("--seq-length", type=int, default=512)

    args = parser.parse_args()
    main(args)
