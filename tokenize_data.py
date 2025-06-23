# build_tokenizer.py
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

# 1. 准备 vocab
special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>", "<mask>"]
aa_tokens = list("ACDEFGHIKLMNPQRSTVWY")        # 20 个标准氨基酸
vocab = {tok: i for i, tok in enumerate(special_tokens + aa_tokens)}

# 2. 用 tokenizers 构造 WordLevel 模型
backend_tok = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
backend_tok.pre_tokenizer = Whitespace()

# 3. 封装成 HF FastTokenizer
hf_tok = PreTrainedTokenizerFast(
    tokenizer_object=backend_tok,
    unk_token="<unk>",
    pad_token="<pad>",
    bos_token="<bos>",
    eos_token="<eos>",
    mask_token="<mask>",
)

print("vocab size:", hf_tok.vocab_size)   # 25
print("mask id:", hf_tok.mask_token_id)   # 4

# 4. 保存
save_dir = "aa_tokenizer"
hf_tok.save_pretrained(save_dir)
