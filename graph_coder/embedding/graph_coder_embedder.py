import pathlib

import fasttext

__allowed_args__ = [
    "input",
    "model",
    "lr",
    "dim",
    "ws",
    "epoch",
    "minCount",
    "minCountLabel",
    "minn",
    "maxn",
    "neg",
    "wordNgrams",
    "loss",
    "bucket",
    "thread",
    "lrUpdateRate",
    "t",
    "label",
    "verbose",
    "pretrainedVectors",
]


class GraphCoderEmbedder:
    def __init__(self, model):
        self.model = model

    @classmethod
    def train_unsupervised(cls, input_file, *args, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k in __allowed_args__}
        model = fasttext.train_unsupervised(input_file, *args, **kwargs)
        return cls(model)

    @classmethod
    def from_pretrained(cls, path):
        return cls(fasttext.load_model(str(pathlib.Path(path).expanduser())))

    def get_word_vector(self, word):
        return self.model.get_word_vector(word)

    def get_sentence_vector(self, text):
        return self.model.get_sentence_vector(text)

    def get_nearest_neighbors(self, word, k=10):
        return self.model.get_nearest_neighbors(word, k=k)

    def get_analogies(self, a, b, c, k=10):
        return self.model.get_analogies(a, b, c, k=k)

    def save(self, path):
        self.model.save_model(str(pathlib.Path(path).expanduser()))
