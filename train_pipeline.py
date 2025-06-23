import argparse
import torch
import os
import json
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from tqdm import tqdm
from model_llama import LlamaForCausalLM
from flash_attn.losses.cross_entropy import CrossEntropyLoss
from peft import LoraConfig, get_peft_model

# 加载自定义 tokenizer
from aa_tokenizer import ProteinCharTokenizer

class AlpacaDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        self.data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        protein = self.data[idx]["input"]
        rna = self.data[idx]["output"]
        text = f"{protein}<sep>{rna}"
        input_ids = self.tokenizer.encode(text, max_length=self.max_length)
        return input_ids

def create_dataloaders(
    batch_size: int,
    block_size: int,
    accelerator,
    train_data_dir: str,
    val_data_dir: Optional[str] = None,
    seed: int = 3407,
) -> tuple[DataLoader, DataLoader]:
    tokenizer = ProteinCharTokenizer()
    train_dataset = AlpacaDataset(train_data_dir, tokenizer, block_size)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
    val_dataloader = None
    return train_dataloader, val_dataloader

def transition(x_0, sigma, maskable_mask, mask_token_id):
    move_chance = sigma
    move_indices = (torch.rand(*x_0.shape, device=x_0.device) < move_chance) & maskable_mask
    x_t = torch.where(move_indices, mask_token_id, x_0)
    return x_t

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulate_every,
        mixed_precision="bf16",
    )
    accelerator.print(f"Total GPUs: {accelerator.num_processes}")

    train_loader, val_dataloader = create_dataloaders(
        batch_size=args.batch_size,
        block_size=args.seq_length,
        accelerator=accelerator,
        train_data_dir=args.dataset,
        seed=3407,
    )

    tokenizer = ProteinCharTokenizer()
    model = LlamaForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
    )
    lora_config = LoraConfig(
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model, optim, train_loader = accelerator.prepare(model, optim, train_loader)
    loss_func = CrossEntropyLoss(inplace_backward=True, reduction='none')

    sampling_eps = 1e-3
    mask_token_id = tokenizer.token2id.get("<mask>", tokenizer.token2id["<unk>"])
    accelerator.print(f"Using mask_token_id: {mask_token_id}")
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    model.train()
    for step, batch in enumerate(train_loader):
        input_ids = batch[..., :args.seq_length]
        target_ids = batch[..., :args.seq_length]
        position_ids = torch.arange(args.seq_length).unsqueeze(0).expand(input_ids.shape[0], -1)
        src_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=input_ids.device)

        t = (1 - sampling_eps) * torch.rand(input_ids.shape[0], device=input_ids.device) + sampling_eps
        sigma = t
        dsigma = torch.reciprocal(t)
        input_ids = transition(input_ids, sigma[:, None], maskable_mask=~src_mask, mask_token_id=mask_token_id)
        loss_mask = input_ids == mask_token_id

        with accelerator.accumulate(model):
            logits = model(input_ids, position_ids=position_ids).logits
            loss = loss_func(
                logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1)
            ).reshape(target_ids.shape[0], -1)
            loss = loss.masked_fill(~loss_mask, 0)
            loss = (dsigma[:, None] * loss).sum() / loss_mask.sum()
            accelerator.backward(loss)
            optim.step()
            optim.zero_grad()

            if accelerator.sync_gradients:
                gathered_loss = accelerator.reduce(loss.clone().detach(), "mean")
                loss_log = {"loss": gathered_loss.item(), "ppl": math.exp(gathered_loss.item())}
                accelerator.log(loss_log, step=completed_steps)
                progress_bar.update(1)
                progress_bar.set_postfix(loss_log)
                completed_steps += 1

        if completed_steps >= args.max_train_steps:
            break

    accelerator.end_training()
    if args.output_dir:
        accelerator.unwrap_model(model).save_pretrained(args.output_dir)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--batch-size", type=int, default=1)
    args.add_argument("--gradient-accumulate-every", type=int, default=4)
    args.add_argument("--output-dir", type=str, required=True)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--max-train-steps", type=int, default=400)
    args.add_argument("--learning-rate", type=float, default=3e-5)
    args.add_argument("--model", type=str, default="/workspace/huangxiaoniu/diffu/DiffuLLaMA/llama-3.2-1B")
    args.add_argument("--dataset", type=str, default="/workspace/huangxiaoniu/diffu/pipeline_data/train_data.jsonl")
    args.add_argument("--seq-length", type=int, default=512)
    args.add_argument("--parallel_mode", type=str, choices=["data_parallel"], default="data_parallel")
    main(args.parse_args())
