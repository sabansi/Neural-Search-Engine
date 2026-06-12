"""Our own WordPiece tokenizer, trained from scratch on our corpora.

The HuggingFace `tokenizers` library is used only as a utility implementation
of the WordPiece training algorithm — the vocabulary itself is learned from
our own data (SQuAD passages + queries and the Jurafsky & Martin book), so no
pretrained artifacts enter the pipeline.
"""

from pathlib import Path

import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


class TextTokenizer:
    """Thin wrapper around a trained `tokenizers.Tokenizer` that produces
    padded PyTorch batches and exposes the special-token ids we need."""

    def __init__(self, tok: Tokenizer):
        self.tok = tok
        self.pad_id = tok.token_to_id("[PAD]")
        self.unk_id = tok.token_to_id("[UNK]")
        self.cls_id = tok.token_to_id("[CLS]")
        self.sep_id = tok.token_to_id("[SEP]")
        self.mask_id = tok.token_to_id("[MASK]")
        self.special_ids = {self.pad_id, self.cls_id, self.sep_id, self.mask_id}

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    @classmethod
    def load(cls, path: str | Path) -> "TextTokenizer":
        path = Path(path)
        if path.is_dir():
            path = path / "tokenizer.json"
        return cls(Tokenizer.from_file(str(path)))

    def save(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.tok.save(str(save_dir / "tokenizer.json"))

    def encode_batch(self, texts: list[str], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a batch, truncate to `max_len`, pad to the longest
        sequence in the batch. Returns (input_ids, attention_mask)."""
        self.tok.enable_truncation(max_length=max_len)
        self.tok.enable_padding(pad_id=self.pad_id, pad_token="[PAD]")
        encs = self.tok.encode_batch(texts)
        input_ids = torch.tensor([e.ids for e in encs], dtype=torch.long)
        attention_mask = torch.tensor([e.attention_mask for e in encs], dtype=torch.long)
        return input_ids, attention_mask


def train_wordpiece(texts: list[str], vocab_size: int = 30_000, save_dir: str | Path | None = None) -> TextTokenizer:
    """Train a BERT-style WordPiece tokenizer from scratch on `texts`."""
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    # Same text normalisation choices as BERT-uncased: NFD unicode
    # normalisation, lowercasing, accent stripping — queries arrive with
    # arbitrary capitalisation, so an uncased vocabulary generalises better.
    tok.normalizer = normalizers.Sequence(
        [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
    )
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        continuing_subword_prefix="##",
    )
    tok.train_from_iterator(texts, trainer=trainer, length=len(texts))

    cls_id = tok.token_to_id("[CLS]")
    sep_id = tok.token_to_id("[SEP]")
    tok.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
    )

    wrapped = TextTokenizer(tok)
    if save_dir is not None:
        wrapped.save(save_dir)
    return wrapped
