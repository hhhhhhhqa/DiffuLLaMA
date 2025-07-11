# train_prot2rna.py
import argparse, json, math, os, random
from typing import List, Dict

import numpy as np
import torch
from accelerate import Accelerator
from flash_attn.losses.cross_entropy import CrossEntropyLoss
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    EncoderDecoderModel,
    PreTrainedTokenizerFast,
    set_seed,
    LlamaConfig,
)
# 显式使用官方实现，避免被本地同名脚本覆盖
from transformers.models.llama import LlamaModel, LlamaForCausalLM

# ----------------------------- 数据集 ----------------------------- #
class Prot2RNADataset(Dataset):
    """一次性把 jsonl 读进内存"""

    def __init__(
        self,
        jsonl_path: str,
        enc_tok: PreTrainedTokenizerFast,
        dec_tok: PreTrainedTokenizerFast,
        max_len_enc: int = 1026,
        max_len_dec: int = 1026,
    ):
        self.enc_tok, self.dec_tok = enc_tok, dec_tok
        self.max_len_enc, self.max_len_dec = max_len_enc, max_len_dec
        with open(jsonl_path, encoding="utf-8") as f:
            self.samples: List[Dict[str, str]] = [json.loads(l) for l in f]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        prot = self.samples[idx]["input"]
        rna  = self.samples[idx]["output"]

        enc = self.enc_tok(
            prot, max_length=self.max_len_enc, truncation=True, padding="max_length",
            add_special_tokens=True, return_tensors="pt",
        )
        dec = self.dec_tok(
            rna,  max_length=self.max_len_dec, truncation=True, padding="max_length",
            add_special_tokens=True, return_tensors="pt",
        )

        dec_input = dec["input_ids"].squeeze(0)
        bos       = torch.tensor([self.dec_tok.bos_token_id])
        decoder_input_ids = torch.cat([bos, dec_input[:-1]], dim=0)

        return {
            "input_ids":        enc["input_ids"].squeeze(0),
            "attention_mask":   enc["attention_mask"].squeeze(0),
            "decoder_input_ids": decoder_input_ids,
            "labels":            dec_input,
        }

# ------------------------- Diffusion 掩码 ------------------------- #
def transition(x_0, sigma, maskable_mask, mask_token_id):
    move_idx = (torch.rand_like(x_0.float()) < sigma) & maskable_mask
    return torch.where(move_idx, mask_token_id, x_0)

# ----------------------------- 主程序 ----------------------------- #
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulate_every,
        mixed_precision="bf16",
    )
    accelerator.print(f"Total GPUs: {accelerator.num_processes}")

    # ------------------ 模型 ------------------ #
    enc = LlamaModel.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
    )

    dec_cfg = LlamaConfig.from_pretrained(args.model)
    dec_cfg.is_decoder = True
    dec_cfg.add_cross_attention = True
    dec_cfg.use_cache = False

    dec = LlamaForCausalLM.from_pretrained(
        args.model,
        config=dec_cfg,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
    )

    # 双保险：确保两个标志位确实打开
    assert dec.config.is_decoder and dec.config.add_cross_attention, \
        "decoder 未正确设置交叉注意力"

    model = EncoderDecoderModel(encoder=enc, decoder=dec)

    # ------------------ Tokenizer ------------------ #
    enc_tok = PreTrainedTokenizerFast.from_pretrained(args.enc_tokenizer)
    dec_tok = PreTrainedTokenizerFast.from_pretrained(args.dec_tokenizer)

    required_enc = ["<unk>", "<pad>", "<bos>", "<eos>"]
    required_dec = ["<unk>", "<pad>", "<bos>", "<eos>", "<mask>"]
    if not all(t in enc_tok.all_special_tokens for t in required_enc):
        raise ValueError("Encoder tokenizer 缺少特殊符号")
    if not all(t in dec_tok.all_special_tokens for t in required_dec):
        raise ValueError("Decoder tokenizer 缺少特殊符号")

    # 调整词表 & 更新 config
    model.encoder.resize_token_embeddings(len(enc_tok))
    model.decoder.resize_token_embeddings(len(dec_tok))
    model.config.update(
        pad_token_id            = dec_tok.pad_token_id,
        decoder_start_token_id  = dec_tok.bos_token_id,
        eos_token_id            = dec_tok.eos_token_id,
        vocab_size              = len(dec_tok),
    )

    accelerator.print(
        dict(enc_vocab=len(enc_tok), dec_vocab=len(dec_tok),
             pad_id=model.config.pad_token_id,
             decoder_bos=model.config.decoder_start_token_id,
             eos_id=model.config.eos_token_id)
    )

    # ------------------ 数据 ------------------ #
    train_ds = Prot2RNADataset(
        args.dataset, enc_tok, dec_tok, args.seq_length, args.seq_length
    )
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, num_workers=4, drop_last=True,
    )

    # ------------------ LoRA ------------------ #
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.gradient_checkpointing_enable()

    # ------------------ Optimizer & prepare ------------------ #
    optim = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
    )
    loss_fn = CrossEntropyLoss(inplace_backward=True, reduction="none")
    model, optim, train_dl = accelerator.prepare(model, optim, train_dl)

    # ------------------ 训练 ------------------ #
    mask_token_id = dec_tok.mask_token_id
    sampling_eps  = 1e-3
    step_done     = 0
    model.train()

    progress = tqdm(range(args.max_train_steps),
                    disable=not accelerator.is_local_main_process)
    for epoch in range(int(np.ceil(args.max_train_steps / len(train_dl)))):
        for batch in train_dl:
            if step_done >= args.max_train_steps:
                break

            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            B, L  = batch["decoder_input_ids"].shape

            sigma  = (1 - sampling_eps) * torch.rand(B, device=batch["input_ids"].device) + sampling_eps
            dsigma = 1.0 / sigma

            special = {dec_tok.bos_token_id, dec_tok.eos_token_id, dec_tok.pad_token_id}
            maskable = torch.ones_like(batch["decoder_input_ids"], dtype=torch.bool)
            for s in special:
                maskable &= ~batch["decoder_input_ids"].eq(s)

            dec_noised = transition(batch["decoder_input_ids"], sigma, maskable, mask_token_id)
            loss_mask  = dec_noised.eq(mask_token_id)

            with accelerator.accumulate(model):
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    decoder_input_ids=dec_noised,
                ).logits

                loss = loss_fn(
                    logits.view(-1, logits.size(-1)),
                    batch["labels"].view(-1),
                ).view(B, L)

                loss = loss.masked_fill(~loss_mask, 0)
                loss = (dsigma.unsqueeze(1) * loss).sum() / loss_mask.sum()

                accelerator.backward(loss)
                optim.step()
                optim.zero_grad()

            if accelerator.sync_gradients:
                loss_val = accelerator.gather(loss.detach()).mean().item()
                progress.update(1)
                progress.set_postfix(loss=f"{loss_val:.4f}",
                                     ppl=f"{math.exp(loss_val):.2f}")
                step_done += 1

    accelerator.end_training()
    accelerator.unwrap_model(model).save_pretrained(args.output_dir)
    enc_tok.save_pretrained(os.path.join(args.output_dir, "enc_tokenizer"))
    dec_tok.save_pretrained(os.path.join(args.output_dir, "dec_tokenizer"))
    accelerator.print("Training finished -- model & tokenizers saved.")

# ----------------------------- CLI ----------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulate-every", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default='/workspace/huangxiaoniu/diffu/DiffuLLaMA/DiffuLLaMA-training')
    parser.add_argument("--max-train-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--model", type=str,
        default="/workspace/huangxiaoniu/diffu/DiffuLLaMA/llama-3.2-1B")
    parser.add_argument("--dataset", type=str,
        default="/workspace/huangxiaoniu/diffu/DiffuLLaMA/DiffuLLaMA-training/train_data.jsonl")
    parser.add_argument("--seq-length", type=int, default=1026)
    parser.add_argument("--enc-tokenizer", type=str,
        default="/workspace/huangxiaoniu/diffu/DiffuLLaMA/DiffuLLaMA-training/protein_tokenizer")
    parser.add_argument("--dec-tokenizer", type=str,
        default="/workspace/huangxiaoniu/diffu/DiffuLLaMA/DiffuLLaMA-training/rna-tokenizer")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
