# inf_diffullama.py
import torch, transformers
from argparse import ArgumentParser
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM
from attention_patch import replace_attention_mask          # 你的 4-D attn patch
from model import DiscreteDiffusionModel, generate_samples  # DiffuLLaMA 框架

replace_attention_mask()   # 必须在 transformers 加载前打 patch

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", default="diffusionfamily/diffullama")
    parser.add_argument("--tokenizer_path", default="./protein_tokenizer")  # ← NEW
    parser.add_argument("--diffusion_steps", type=int, default=64)
    parser.add_argument("--shift", action="store_true", default=True)
    parser.add_argument("--logits_temp",  type=float, default=0.9)
    parser.add_argument("--topp_temp",    type=float, default=0.9)
    parser.add_argument("--verbose",      action="store_true")
    parser.add_argument("--flash_attn",   choices=["eager","sdpa","flash_attention_2"],
                        default="eager")
    return parser.parse_args()

def load_llama_and_resize(args):
    # 1) 载 tokenizer（自定义）
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,         # 如果是自定义 tokenizer class
        model_max_length=2048,
    )

    # 2) 载 config，同时把 vocab_size 改成 tokenizer 长度
    config = AutoConfig.from_pretrained(args.model_name)
    vocab_sz = len(tokenizer)
    if vocab_sz != config.vocab_size:
        print(f"[DiffuLLaMA] Resize vocab: {config.vocab_size} → {vocab_sz}")
        config.vocab_size = vocab_sz

    # 3) 载基座模型
    model = LlamaForCausalLM.from_pretrained(
        args.model_name,
        config=config,                           # ← 传入新的 vocab_size
        device_map="auto",
        _attn_implementation=args.flash_attn,
        torch_dtype=torch.bfloat16,
    )

    # 4) 调整 embedding / lm_head
    #    pad_to_multiple_of=8 可避免 Flash-Attn 的 shape 对齐问题
    model.resize_token_embeddings(vocab_sz, pad_to_multiple_of=8)

    return model, tokenizer, config

def main():
    args = get_args()
    llama, tokenizer, config = load_llama_and_resize(args)

    # 包装成 DiffuLLaMA 的离散扩散模型
    ddm = DiscreteDiffusionModel(
        model=llama,
        config=config,
        tokenizer=tokenizer,
        device="cuda",
    ).cuda()

    # ======= 下面示例推理流程不变 =======
    gen_len = 2048
    print("="*20, "Unconditional generation")
    x0 = [tokenizer.pad_token_id] * gen_len
    inputs = {"input_ids": torch.tensor([x0]).cuda()}
    res = generate_samples(ddm, args, tokenizer, inputs, verbose=args.verbose)
    print(tokenizer.decode(res.tolist()[0], skip_special_tokens=True))

    print("="*20, "Prefix generation")
    prefix_ids = tokenizer.encode("MENDEL", add_special_tokens=True)  # 你的前缀
    x0 = prefix_ids + [tokenizer.pad_token_id]*(gen_len-len(prefix_ids))
    src_mask = [1]*len(prefix_ids) + [0]*(gen_len-len(prefix_ids))
    inputs = {
        "input_ids": torch.tensor([x0]).cuda(),
        "src_mask": torch.tensor([src_mask]).cuda()
    }
    res = generate_samples(ddm, args, tokenizer, inputs, verbose=args.verbose)
    print(tokenizer.decode(res.tolist()[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()

