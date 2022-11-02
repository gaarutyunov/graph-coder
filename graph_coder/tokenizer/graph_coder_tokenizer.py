from tokenizers import Tokenizer, processors, pre_tokenizers, decoders, normalizers
from tokenizers.models import BPE

from graph_coder.tokenizer.base import AbstractTokenizer


class GraphCoderTokenizer(AbstractTokenizer):
    def __init__(self, tokenizer: Tokenizer = Tokenizer(BPE(unk_token="[UNK]"))):
        super().__init__(tokenizer)
        self.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.decoder = decoders.ByteLevel()
        self.post_processor = processors.ByteLevel(trim_offsets=False)
        self.normalizer = normalizers.NFKC()
