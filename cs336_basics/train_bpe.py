"""
Problem (train_bpe):
BPE Tokenizer Training (15 points)

Deliverable: Write a function that, given a path to an input text file, trains a
(byte-level) BPE tokenizer. Your BPE training function should handle (at least) the
following input parameters:
- input_path: str Path to a text file with BPE tokenizer training data.
- vocab_size: int A positive integer that defines the maximum final vocabulary size
  (including the initial byte vocabulary, vocabulary items produced from merging, and
  any special tokens).
- special_tokens: list[str] A list of strings to add to the vocabulary. These special
tokens do not otherwise affect BPE training.

Your BPE training function should return the resulting vocabulary and merges:
vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the
vocabulary) to bytes (token bytes).
merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each
list item is a tuple of bytes (<token1>, <token2>), representing that <token1> was
merged with <token2>. The merges should be ordered by order of creation.

To test your BPE training function against our provided tests, you will first need to
implement the test adapter at [adapters.run_train_bpe]. Then, run
`uv run pytest tests/test_train_bpe.py`.
Your implementation should be able to pass all tests. Optionally (this could be a large
time-investment), you can implement the key parts of your training method using some
systems language, for instance C++ (consider cppyy for this)or Rust (using PyO3). If you
do this, be aware of which operations require copying vs reading directly from Python
memory, and make sure to leave build instructions, or make sure it builds using only
pyproject.toml. Also note that the GPT-2 regex is not well-supported in most regex
engines and will be too slow in most that do. We have verified that Oniguruma is
reasonably fast and supports negative lookahead, but the regex package in Python is, if
anything, even faster.

It maybe helpful to look at
- tests/fixtures/train-bpe-reference-merges.txt
- tests/fixtures/train-bpe-reference-vocab.json
to see sample output from a method like this.


> uv run cs336_basics/train_bpe.py
> uv run pytest

2026-04-07

The BPE going from BPE'd integers to unicode characters and sequences of them is simple.
Each integer maps to some string of unicode characters, and the BPE merges are just a
list of pairs of strings that were merged together.

Going the other ways used to confuse me ... becuase, if I encode, for example, "at" as
257, and "cat" as 1049, then how would i know how to encode "scatalogical"?

The trick is that you learned the extra tokens one at a time, by studying the training
data and observing the most common pairs of integers.  So when performing the BPE
mapping from the text to the integers, we start with simple unicode ("step zero") and
then iterate. In the next step we would apply the first learned pairing (such as "a + t"
 is really common) and then, in the next step , ... and so on.

It may be required to  store the training data as numpy arrays.

Note on collections.Counter: A Counter is a dict subclass for counting hashable objects.
It is an unordered collection where elements are stored as dictionary keys and their
counts are stored as dictionary values. Counts are allowed to be any integer value
including zero or negative counts. The Counter class is a part of the collections
module in Python's standard library and provides convenient methods for counting and
manipulating counts of objects.
Primary Function: It automatically tallies the frequency of elements in an iterable
(like a list or string) without requiring manual loops.
"""

import pathlib
from collections import Counter
from itertools import chain
# from loguru import logger
from typing import Literal, Optional, Union

import regex

from cs336_basics import CS336_BASICS_ROOT

# from functools import lru_cache, cache




DATA_FOLDER = CS336_BASICS_ROOT.joinpath("data")
print(f"data folder = {DATA_FOLDER}")
assert DATA_FOLDER.exists()
DEFAULT_SPECIAL_TOKENS = [
    "<|endoftext|>",
    "qokka",
]  # [ b"<unk>", b"<pad>", b"<s>", b"</s>", ]  # noqa E501
TOY_INPUT_FILE = DATA_FOLDER.joinpath("toy_string.txt")
DEFAULT_PRETOKENIZE_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""  # noqa E501
DEFAULT_PRETOKENIZE_REGEX = regex.compile(DEFAULT_PRETOKENIZE_PATTERN)


def make_initial_vocab(debug: bool = False) -> dict[int, bytes]:
    """
    Create the initial vocabulary mapping each byte value (0-255) to its corresponding
    byte representation.

    This example taken from Ron's
    https://github.com/rmayer-sst/stanford-cs336-assignment1-basics/blob/ron/cs336_basics/ron_train_bpe.py  # noqa E501

    Parameters:
    -----------
    debug: bool
        If True, print the initial vocabulary for debugging purposes.

    Returns:
    --------
    dict[int, bytes]
        A dictionary mapping each byte value to its corresponding byte representation.
    """
    vocab = {i: bytes([i]) for i in range(256)}
    if debug:
        print(f"Initial vocab: {vocab}")
        for k, v in vocab.items():
            print(f"{k}: {v}, {v.decode('utf-8', errors='replace')}")
    return vocab


def string_as_byte_list(s: str) -> list[int]:
    """
    Convert a string to a list of byte values (integers in the range 0-255) using UTF-8
    encoding.

    for example, the string "hello" would be converted to [104, 101, 108, 108, 111],
    since those are UTF-8 byte values for the characters 'h', 'e', 'l', 'l', and 'o'.

    Parameters:
    -----------
    s: str
        The input string to be converted to a list of byte values.  This string will be
        encoded using UTF-8, which can represent all Unicode characters.

    Returns:
    --------
    list[int]
        A list of byte values representing the input string.
    """
    return [b for b in s.encode("utf-8")]


# def count_token_pairs(tokids: list[int]) -> dict[tuple[int, int], int]:
#     """
#     Count the frequency of adjacent token pairs in a list of token IDs.

#     Parameters:
#     -----------
#     tokids: list[int]
#         A list of token IDs representing a sequence of tokens.
#     Returns:
#     --------
#     dict[tuple[int, int], int]
#         A dictionary where the keys are tuples of adjacent token IDs (token pairs) and the
#         values are the counts of how many times each token pair appears in the input list.
#     Example:
#     --------
#     >>> count_token_pairs([1, 2, 3, 2, 1, 2])
#     {(1, 2): 2, (2, 3): 1, (3, 2): 1, (2, 1): 1}
#     """
#     cnts = {}
#     for a, b in zip(tokids, tokids[1:]):
#         cnts[(a, b)] = cnts.get((a, b), 0) + 1
#     return cnts


# # @lru_cache(maxsize=10000)
# @cache
# def count_token_pairs_cached(tokids_tuple: tuple[int, ...]):
#     return count_token_pairs(list(tokids_tuple))


def train_bpe(
    input_path: Union[str, pathlib.Path] = TOY_INPUT_FILE,
    vocab_size: int = 256 * 10,
    special_tokens: list[str] = DEFAULT_SPECIAL_TOKENS,
    debug: bool = False,
):
    """

    Parameters:
    -----------
    input_path: Union[str, pathlib.Path]
        text file with BPE tokenizer training data.
    vocab_size: int
        A positive integer that defines the maximum final vocabulary size.
    special_tokens: list[str]
        A list of strings to add to the vocabulary. These special tokens do not
        otherwise affect BPE training.

    Returns:
    -------
    vocab: dict[int, bytes]
        The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to
        bytes (token bytes).
    merges: list[tuple[bytes, bytes]]
        A list of BPE merges produced from training. Each list item is a tuple of bytes
        (<token1>, <token2>),representing that <token1> was merged with <token2>.
        The merges should be ordered by order of creation.

    Development Notes:
    Main Steps:
        - special tokens splits (e.g. remove weird characters that are used to navigate
        the document, not communicate language meaning)
        - Pretokenize (Use re.finditer() and read the docs)
        - Convert pretokenized data into UTF-8 Bytes
        - Train BPE merges on the byte-level representation

    Pretokenization:
    bascially boils down to:
    pretokens = []
    for segment in segments:
        for pt in pretokenize(segment):
            pretokens.append(pt)
    but you can also do it in a more compact way using list comprehensions, like in
    Ron's code.  Here we use an equivalent way of writing the same thing, but using
    itertools.chain.from_iterable.
    So, if segments is ["hello world", "foo bar"], and pretokenize splits on spaces,
    you get:
    pretokenize("hello world") → ["hello", "world"]
    pretokenize("foo bar") → ["foo", "bar"]
    The double for loop flattens these into one list: ["hello", "world", "foo", "bar"]

    """
    vocab = make_initial_vocab()
    merges = []
    new_vocab_idx = len(vocab)

    # add special tokens to vocab
    for st in special_tokens:
        vocab[new_vocab_idx] = st.encode("utf-8")
        new_vocab_idx += 1

    # get input text
    with open(input_path) as f:
        text = f.read()
    if debug:
        print(f"Original text\n {text}")

    # Apply special splits
    segments = split_on_special_tokens(
        input_string=text, special_tokens=special_tokens, rejoin=False
    )

    # Pretokenize
    pretokenized_text = list(
        chain.from_iterable(pretokenize(segment) for segment in segments)
    )
    if debug:
        print(f"pretokenized_text\n {pretokenized_text}")

    # Now that you have the pretokenized text, you can convert it to bytes and then
    # train the BPE merges on the byte-level representation.

    # TODO: Add frequency counter tools (see Ron's code) to count the frequency of
    # token pairs
    pretok_freqs = dict(Counter(pretokenized_text))
    pretok_deduped, pretok_weights = zip(*pretok_freqs.items())
    pretok_deduped = list(pretok_deduped)
    pretok_weights = list(pretok_weights)
    pretok_ids = [string_as_byte_list(pt) for pt in pretok_deduped]
    # n_pretok = len(pretok_deduped)
    # uncomment when using toy example to see the frequencies of the pretokenized text
    if debug:
        print(f"pretok_freqs\n {pretok_freqs}")
        print(f"pretok_deduped\n {pretok_deduped}")
        print(f"pretok_weights\n {pretok_weights}")
        print(f"pretok_ids\n {pretok_ids}")

    # main BPE training loop..

    # # initialize counts of token pairs
    # counts = Counter()
    # pretok_batch_counts = [
    #     count_token_pairs_cached(tuple(tokids)) for tokids in pretok_ids
    # ]
    # for bc, wt in zip(pretok_batch_counts, pretok_weights):
    #     for k, v in bc.items():
    #         counts[k] += v * wt

    # new_style_counts = None
    # for i_merge in range(vocab_size - 257):
    #     print(f"merge {i_merge}")
    #     best_pair = get_best_pair(counts, vocab)
    #     best_bytes = tokids_to_bytestring(best_pair, vocab)
    #     vocab[new_vocab_idx] = best_bytes
    #     # this_merge = tuple([vocab[tid] for tid in best_pair])
    #     this_merge = (vocab[best_pair[0]], vocab[best_pair[1]])  # slightly faster
    #     merges.append(this_merge)
    #     # print(f"merged {this_merge}")
    #     merge_results = [
    #         merge_tokids(tokids, best_pair, new_vocab_idx) for tokids in pretok_ids
    #     ]
    #     pretok_ids = [newtoks for newtoks, deltas in merge_results]
    #     deltas = [deltas for newtoks, deltas in merge_results]
    #     for d, wt in zip(deltas, pretok_weights):
    #         if d:
    #             for k, v in d.items():
    #                 counts[k] += v * wt

    #     new_vocab_idx += 1
    #     # print(f"####### {pretokids}")

    print("lv", len(vocab), "lm", len(merges))
    return vocab, merges
    # bytes =

    return


def split_on_special_tokens(
    input_string: str, special_tokens: list[str], rejoin: bool = True
) -> Union[list[str], str]:
    """
    Split a string on special tokens.

    Parameters
    ----------
    input_string : str
        The input string to be split.
    special_tokens : list of str or str
        Special tokens to split the input string on. Each occurrence of a special token
        will be used as a split point.
    rejoin : bool, optional
        If True, the resulting list of strings will be joined into a single string
        separated by spaces.
        If False, returns a list of split strings. Default is True.

    Returns
    -------
    splitted: Union[list[str], str]
        The input string split on the special tokens. If `rejoin` is True, returns a
        single string with the split parts joined by spaces.
        Note the split operation (string chunking) will drop all the special_tokens.


    Examples
    --------
    >>> split_on_special_tokens("Hello <|endoftext|> world", ["<|endoftext|>"])
    ['Hello  world']

    >>> split_on_special_tokens("foo bar baz", ["bar"], rejoin=False)
    ['foo ', ' baz']

    """

    # sorted_tokens = special_tokens.sort()
    pattern = "|".join(regex.escape(st) for st in special_tokens)
    splitted = regex.split(pattern=pattern, string=input_string)
    if rejoin:
        return " ".join(splitted)

    return splitted


def pretokenize(
    text: str,
    regex_pattern: Optional[str] = DEFAULT_PRETOKENIZE_REGEX,
    method: Literal["regex_findall", "regex_finditer"] = "regex_findall",
) -> list[str]:
    """
    Parameters
    ----------
    text : str
        The input string to be pretokenized.
    regex_pattern : str, optional
        The regular expression pattern to use for pretokenization.
        Default is a pattern that matches common English contractions, letters, numbers,
        and punctuation.
    method : Literal["regex_findall", "regex_finditer"], optional
        The method to use for pretokenization. Default is "regex_findall".

    Returns
    -------
    list of str
        The pretokenized strings.

        >>> # requires `regex` package
        >>> import regex as re
        >>> re.findall(PAT, "some text that i'll pre-tokenize")
        ['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']


    TODO: review usage of 'match', 'pattern', 'search' methods of
    <class '_regex.Scanner'>

    """
    if method == "regex_findall":
        # output = regex.findall(regex_pattern, text)
        output = regex_pattern.findall(text)
    elif method == "regex_finditer":
        # output = [match.group(0) for match in regex.finditer(regex_pattern, text)]
        output = [match.group(0) for match in regex_pattern.finditer(text)]
    else:
        raise ValueError(
            f"Invalid method: {method}. Must be 'regex_findall' or 'regex_finditer'."
        )
    return output


def main():
    train_bpe(
        input_path=TOY_INPUT_FILE,
        vocab_size=256 * 10,
        special_tokens=DEFAULT_SPECIAL_TOKENS,
        debug=True,
    )


if __name__ == "__main__":
    main()
