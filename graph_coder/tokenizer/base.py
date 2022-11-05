from abc import ABC

from tokenizers import Tokenizer


class TokenizerBase(ABC):
    """
    A :obj:`Tokenizer` works as a pipeline. It processes some raw text as input
    and outputs an :class:`~tokenizers.Encoding`.

    Args:
        model (:class:`~tokenizers.models.Model`):
            The core algorithm that this :obj:`Tokenizer` should be using.

    """

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def add_special_tokens(self, tokens):
        """
        Add the given special tokens to the Tokenizer.

        If these tokens are already part of the vocabulary, it just let the Tokenizer know about
        them. If they don't exist, the Tokenizer creates them, giving them a new id.

        These special tokens will never be processed by the model (ie won't be split into
        multiple tokens), and they can be removed from the output when decoding.

        Args:
            tokens (A :obj:`List` of :class:`~tokenizers.AddedToken` or :obj:`str`):
                The list of special tokens we want to add to the vocabulary. Each token can either
                be a string or an instance of :class:`~tokenizers.AddedToken` for more
                customization.

        Returns:
            :obj:`int`: The number of tokens that were created in the vocabulary
        """
        self.tokenizer.add_special_tokens(tokens)

    def add_tokens(self, tokens):
        """
        Add the given tokens to the vocabulary

        The given tokens are added only if they don't already exist in the vocabulary.
        Each token then gets a new attributed id.

        Args:
            tokens (A :obj:`List` of :class:`~tokenizers.AddedToken` or :obj:`str`):
                The list of tokens we want to add to the vocabulary. Each token can be either a
                string or an instance of :class:`~tokenizers.AddedToken` for more customization.

        Returns:
            :obj:`int`: The number of tokens that were created in the vocabulary
        """
        self.tokenizer.add_tokens(tokens)

    def decode(self, ids, skip_special_tokens=True):
        """
        Decode the given list of ids back to a string

        This is used to decode anything coming back from a Language Model

        Args:
            ids (A :obj:`List/Tuple` of :obj:`int`):
                The list of ids that we want to decode

            skip_special_tokens (:obj:`bool`, defaults to :obj:`True`):
                Whether the special tokens should be removed from the decoded string

        Returns:
            :obj:`str`: The decoded string
        """
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(self, sequences, skip_special_tokens=True):
        """
        Decode a batch of ids back to their corresponding string

        Args:
            sequences (:obj:`List` of :obj:`List[int]`):
                The batch of sequences we want to decode

            skip_special_tokens (:obj:`bool`, defaults to :obj:`True`):
                Whether the special tokens should be removed from the decoded strings

        Returns:
            :obj:`List[str]`: A list of decoded strings
        """
        return self.tokenizer.decode(sequences, skip_special_tokens=skip_special_tokens)

    @property
    def decoder(self):
        """
        The `optional` :class:`~tokenizers.decoders.Decoder` in use by the Tokenizer
        """
        return self.tokenizer.decoder

    @decoder.setter
    def decoder(self, decoder):
        """
        Set the `optional` :class:`~tokenizers.decoders.Decoder` to use by the Tokenizer

        Args:
            decoder (:class:`~tokenizers.decoders.Decoder`):
                The decoder to use
        """
        self.tokenizer.decoder = decoder

    def enable_padding(
        self,
    ):
        """
        Enable the padding
        """
        self.tokenizer.enable_padding()

    def enable_truncation(
        self, max_length, stride=0, strategy="longest_first", direction="right"
    ):
        """
        Enable truncation

        Args:
            max_length (:obj:`int`):
                The max length at which to truncate

            stride (:obj:`int`, `optional`):
                The length of the previous first sequence to be included in the overflowing
                sequence

            strategy (:obj:`str`, `optional`, defaults to :obj:`longest_first`):
                The strategy used to truncation. Can be one of ``longest_first``, ``only_first`` or
                ``only_second``.

            direction (:obj:`str`, defaults to :obj:`right`):
                Truncate direction
        """
        self.tokenizer.enable_truncation(max_length, stride, strategy, direction)

    def encode(
        self, sequence, pair=None, is_pretokenized=False, add_special_tokens=True
    ):
        """
        Encode the given sequence and pair. This method can process raw text sequences
        as well as already pre-tokenized sequences.

        Example:
            Here are some examples of the inputs that are accepted::

                encode("A single sequence")`
                encode("A sequence", "And its pair")`
                encode([ "A", "pre", "tokenized", "sequence" ], is_pretokenized=True)`
                encode(
                    [ "A", "pre", "tokenized", "sequence" ], [ "And", "its", "pair" ],
                    is_pretokenized=True
                )

        Args:
            sequence (:obj:`~tokenizers.InputSequence`):
                The main input sequence we want to encode. This sequence can be either raw
                text or pre-tokenized, according to the ``is_pretokenized`` argument:

                - If ``is_pretokenized=False``: :class:`~tokenizers.TextInputSequence`
                - If ``is_pretokenized=True``: :class:`~tokenizers.PreTokenizedInputSequence`

            pair (:obj:`~tokenizers.InputSequence`, `optional`):
                An optional input sequence. The expected format is the same that for ``sequence``.

            is_pretokenized (:obj:`bool`, defaults to :obj:`False`):
                Whether the input is already pre-tokenized

            add_special_tokens (:obj:`bool`, defaults to :obj:`True`):
                Whether to add the special tokens

        Returns:
            :class:`~tokenizers.Encoding`: The encoded result

        """
        return self.tokenizer.encode(
            sequence, pair, is_pretokenized, add_special_tokens
        )

    def encode_batch(self, input, is_pretokenized=False, add_special_tokens=True):
        """
        Encode the given batch of inputs. This method accept both raw text sequences
        as well as already pre-tokenized sequences.

        Example:
            Here are some examples of the inputs that are accepted::

                encode_batch([
                    "A single sequence",
                    ("A tuple with a sequence", "And its pair"),
                    [ "A", "pre", "tokenized", "sequence" ],
                    ([ "A", "pre", "tokenized", "sequence" ], "And its pair")
                ])

        Args:
            input (A :obj:`List`/:obj:`Tuple` of :obj:`~tokenizers.EncodeInput`):
                A list of single sequences or pair sequences to encode. Each sequence
                can be either raw text or pre-tokenized, according to the ``is_pretokenized``
                argument:

                - If ``is_pretokenized=False``: :class:`~tokenizers.TextEncodeInput`
                - If ``is_pretokenized=True``: :class:`~tokenizers.PreTokenizedEncodeInput`

            is_pretokenized (:obj:`bool`, defaults to :obj:`False`):
                Whether the input is already pre-tokenized

            add_special_tokens (:obj:`bool`, defaults to :obj:`True`):
                Whether to add the special tokens

        Returns:
            A :obj:`List` of :class:`~tokenizers.Encoding`: The encoded batch

        """
        return self.tokenizer.encode_batch(input, is_pretokenized, add_special_tokens)

    @classmethod
    def from_buffer(cls, buffer):
        """
        Instantiate a new :class:`~tokenizers.Tokenizer` from the given buffer.

        Args:
            buffer (:obj:`bytes`):
                A buffer containing a previously serialized :class:`~tokenizers.Tokenizer`

        Returns:
            :class:`~tokenizers.Tokenizer`: The new tokenizer
        """
        return cls(Tokenizer.from_buffer(buffer))

    @classmethod
    def from_file(cls, path):
        """
        Instantiate a new :class:`~tokenizers.Tokenizer` from the file at the given path.

        Args:
            path (:obj:`str`):
                A path to a local JSON file representing a previously serialized
                :class:`~tokenizers.Tokenizer`

        Returns:
            :class:`~tokenizers.Tokenizer`: The new tokenizer
        """
        return cls(Tokenizer.from_file(path))

    @classmethod
    def from_pretrained(cls, identifier, revision="main", auth_token=None):
        """
        Instantiate a new :class:`~tokenizers.Tokenizer` from an existing file on the
        Hugging Face Hub.

        Args:
            identifier (:obj:`str`):
                The identifier of a Model on the Hugging Face Hub, that contains
                a tokenizer.json file
            revision (:obj:`str`, defaults to `main`):
                A branch or commit id
            auth_token (:obj:`str`, `optional`, defaults to `None`):
                An optional auth token used to access private repositories on the
                Hugging Face Hub

        Returns:
            :class:`~tokenizers.Tokenizer`: The new tokenizer
        """
        return cls(Tokenizer.from_pretrained(identifier, revision, auth_token))

    @classmethod
    def from_str(cls, json):
        """
        Instantiate a new :class:`~tokenizers.Tokenizer` from the given JSON string.

        Args:
            json (:obj:`str`):
                A valid JSON string representing a previously serialized
                :class:`~tokenizers.Tokenizer`

        Returns:
            :class:`~tokenizers.Tokenizer`: The new tokenizer
        """
        return cls(Tokenizer.from_str(json))

    def get_vocab(self, with_added_tokens=True):
        """
        Get the underlying vocabulary

        Args:
            with_added_tokens (:obj:`bool`, defaults to :obj:`True`):
                Whether to include the added tokens

        Returns:
            :obj:`Dict[str, int]`: The vocabulary
        """
        return self.tokenizer.get_vocab(with_added_tokens)

    def get_vocab_size(self, with_added_tokens=True):
        """
        Get the size of the underlying vocabulary

        Args:
            with_added_tokens (:obj:`bool`, defaults to :obj:`True`):
                Whether to include the added tokens

        Returns:
            :obj:`int`: The size of the vocabulary
        """
        return self.tokenizer.get_vocab_size(with_added_tokens)

    def id_to_token(self, id):
        """
        Convert the given id to its corresponding token if it exists

        Args:
            id (:obj:`int`):
                The id to convert

        Returns:
            :obj:`Optional[str]`: An optional token, :obj:`None` if out of vocabulary
        """
        return self.tokenizer.id_to_token(id)

    @property
    def model(self):
        """
        The :class:`~tokenizers.models.Model` in use by the Tokenizer
        """
        return self.tokenizer.model

    @model.setter
    def model(self, model):
        """
        Set the :class:`~tokenizers.models.Model` to use by the Tokenizer

        Args:
            model (:class:`~tokenizers.models.Model`):
                The model to use
        """
        self.tokenizer.model = model

    def no_padding(self):
        """
        Disable padding
        """
        self.tokenizer.no_padding()

    def no_truncation(self):
        """
        Disable truncation
        """
        self.tokenizer.no_truncation()

    @property
    def normalizer(self):
        """
        The `optional` :class:`~tokenizers.normalizers.Normalizer` in use by the Tokenizer
        """
        return self.tokenizer.normalizer

    @normalizer.setter
    def normalizer(self, normalizer):
        """
        The `optional` :class:`~tokenizers.normalizers.Normalizer` in use by the Tokenizer
        """
        self.tokenizer.normalizer = normalizer

    def num_special_tokens_to_add(self, is_pair):
        """
        Return the number of special tokens that would be added for single/pair sentences.
        :param is_pair: Boolean indicating if the input would be a single sentence or a pair
        :return:
        """
        return self.tokenizer.num_special_tokens_to_add(is_pair)

    @property
    def padding(self):
        """
        Get the current padding parameters

        `Cannot be set, use` :meth:`~tokenizers.Tokenizer.enable_padding` `instead`

        Returns:
            (:obj:`dict`, `optional`):
                A dict with the current padding parameters if padding is enabled
        """
        return self.tokenizer.padding

    def post_process(self, encoding, pair=None, add_special_tokens=True):
        """
        Apply all the post-processing steps to the given encodings.

        The various steps are:

            1. Truncate according to the set truncation params (provided with
               :meth:`~tokenizers.Tokenizer.enable_truncation`)
            2. Apply the :class:`~tokenizers.processors.PostProcessor`
            3. Pad according to the set padding params (provided with
               :meth:`~tokenizers.Tokenizer.enable_padding`)

        Args:
            encoding (:class:`~tokenizers.Encoding`):
                The :class:`~tokenizers.Encoding` corresponding to the main sequence.

            pair (:class:`~tokenizers.Encoding`, `optional`):
                An optional :class:`~tokenizers.Encoding` corresponding to the pair sequence.

            add_special_tokens (:obj:`bool`):
                Whether to add the special tokens

        Returns:
            :class:`~tokenizers.Encoding`: The final post-processed encoding
        """
        self.tokenizer.post_process(encoding, pair, add_special_tokens)

    @property
    def post_processor(self):
        """
        The `optional` :class:`~tokenizers.processors.PostProcessor` in use by the Tokenizer
        """
        return self.tokenizer.post_processor

    @post_processor.setter
    def post_processor(self, processor):
        """
        The `optional` :class:`~tokenizers.processors.PostProcessor` in use by the Tokenizer
        """
        self.tokenizer.post_processor = processor

    @property
    def pre_tokenizer(self):
        """
        The `optional` :class:`~tokenizers.pre_tokenizers.PreTokenizer` in use by the Tokenizer
        """
        return self.tokenizer.pre_tokenizer

    @pre_tokenizer.setter
    def pre_tokenizer(self, pre_tokenizer):
        """
        The `optional` :class:`~tokenizers.pre_tokenizers.PreTokenizer` in use by the Tokenizer
        """
        self.tokenizer.pre_tokenizer = pre_tokenizer

    def save(self, path, pretty=True):
        """
        Save the :class:`~tokenizers.Tokenizer` to the file at the given path.

        Args:
            path (:obj:`str`):
                A path to a file in which to save the serialized tokenizer.

            pretty (:obj:`bool`, defaults to :obj:`True`):
                Whether the JSON file should be pretty formatted.
        """
        self.tokenizer.save(path, pretty)

    def to_str(self, pretty=False):
        """
        Gets a serialized string representing this :class:`~tokenizers.Tokenizer`.

        Args:
            pretty (:obj:`bool`, defaults to :obj:`False`):
                Whether the JSON string should be pretty formatted.

        Returns:
            :obj:`str`: A string representing the serialized Tokenizer
        """
        self.tokenizer.to_str(pretty)

    def token_to_id(self, token):
        """
        Convert the given token to its corresponding id if it exists

        Args:
            token (:obj:`str`):
                The token to convert

        Returns:
            :obj:`Optional[int]`: An optional id, :obj:`None` if out of vocabulary
        """
        return self.tokenizer.token_to_id(token)

    def train(self, files, trainer=None):
        """
        Train the Tokenizer using the given files.

        Reads the files line by line, while keeping all the whitespace, even new lines.
        If you want to train from data store in-memory, you can check
        :meth:`~tokenizers.Tokenizer.train_from_iterator`

        Args:
            files (:obj:`List[str]`):
                A list of path to the files that we should use for training

            trainer (:obj:`~tokenizers.trainers.Trainer`, `optional`):
                An optional trainer that should be used to train our Model
        """
        self.tokenizer.train(files, trainer)

    def train_from_iterator(self, iterator, trainer=None, length=None):
        """
        Train the Tokenizer using the provided iterator.

        You can provide anything that is a Python Iterator

            * A list of sequences :obj:`List[str]`
            * A generator that yields :obj:`str` or :obj:`List[str]`
            * A Numpy array of strings
            * ...

        Args:
            iterator (:obj:`Iterator`):
                Any iterator over strings or list of strings

            trainer (:obj:`~tokenizers.trainers.Trainer`, `optional`):
                An optional trainer that should be used to train our Model

            length (:obj:`int`, `optional`):
                The total number of sequences in the iterator. This is used to
                provide meaningful progress tracking
        """
        self.tokenizer.train_from_iterator(iterator, trainer, length)

    @property
    def truncation(self):
        """
        Get the currently set truncation parameters

        `Cannot set, use` :meth:`~tokenizers.Tokenizer.enable_truncation` `instead`

        Returns:
            (:obj:`dict`, `optional`):
                A dict with the current truncation parameters if truncation is enabled
        """
        return self.tokenizer.truncation
