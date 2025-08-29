"""
Problem (train_bpe): 
BPE Tokenizer Training (15 points)  

Deliverable: Write a function that, given a path to an input text file, trains a (byte-level) BPE tokenizer. 
Your BPE training function should handle (at least) the following input parameters:  
- input_path: str Path to a text file with BPE tokenizer training data.  
- vocab_size: int A positive integer that defines the maximum final vocabulary size 
  (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).  
- special_tokens: list[str] A list of strings to add to the vocabulary. These special tokens do not 
otherwise affect BPE training.  

Your BPE training function should return the resulting vocabulary and merges:  
vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item is a tuple of 
bytes (<token1>, <token2>), representing that <token1> was merged with <token2>. The merges should be ordered
 by order of creation.  
 
 To test your BPE training function against our provided tests, you will first need to implement the test 
 adapter at [adapters.run_train_bpe]. Then, run uv run pytest tests/test_train_bpe.py. Your implementation 
 should be able to pass all tests. Optionally (this could be a large time-investment), you can implement 
 the key parts of your training method using some systems language, for instance C++ (consider cppyy for this) 
 or Rust (using PyO3). If you do this, be aware of which operations require copying vs reading directly from 
 Python memory, and make sure to leave build instructions, or make sure it builds using only pyproject.toml. 
 Also note that the GPT-2 regex is not well-supported in most regex engines and will be too slow in most that 
 do. We have verified that Oniguruma is reasonably fast and supports negative lookahead, but the regex 
 package in Python is, if anything, even faster.

 It maybe helpful to look at 
 tests/fixtures/train-bpe-reference-merges.txt
tests/fixtures/train-bpe-reference-vocab.json
to see sample output from a method like this.


> uv run problems/train_bpe.py
> uv run pytest

===================================== 46 failed, 2 skipped, 1 warning in 19.85s =====================================

"""

# from loguru import logger
from typing import Optional, Union

import pathlib
import regex


ASSIGNEMNT_FOLDER = pathlib.Path.home().joinpath("software/sound_thinking/stanford_cs/cs336-assignment1-basics")
DATA_FOLDER = ASSIGNEMNT_FOLDER.joinpath("data")
print(f"data folder = {DATA_FOLDER}")
assert DATA_FOLDER.exists()
DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>", "qokka"]  # [ b"<unk>", b"<pad>", b"<s>", b"</s>", ]
TOY_INPUT_FILE = DATA_FOLDER.joinpath("toy_string.txt")

def train_bpe(
        input_path: Union[str, pathlib.Path] = TOY_INPUT_FILE,
        vocab_size: int = 256 * 10,
        special_tokens: list[str] = DEFAULT_SPECIAL_TOKENS,
):
    """

    Parameters:
    -----------
    input_path: Union[str, pathlib.Path]
        text file with BPE tokenizer training data.
    vocab_size: int
        A positive integer that defines the maximum final vocabulary size.
    special_tokens: list[str] 
        A list of strings to add to the vocabulary. These special tokens do not otherwise affect BPE training.  

    Returns:
    -------
    vocab: dict[int, bytes] 
        The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
    merges: list[tuple[bytes, bytes]] 
        A list of BPE merges produced from training. Each list item is a tuple of bytes (<token1>, <token2>), 
        representing that <token1> was merged with <token2>. The merges should be ordered by order of creation.  

    Development Notes:
    Main Steps:
        - special tokens splits (e.g. remove weird characters that are used to navigate document, not communicate language meaning)
        - Pretokenize (Use re.finditer() and read the docs)
        - Convert pretokenized data into UTF-8 Bytes
        - 
    """
    # get input text
    with open(input_path) as f:
        text = f.read()
    print(f"Original text\n {text}")
    
    # Apply special splits
    splitted_text = split_on_special_tokens(
        input_string=text,
        special_tokens=special_tokens
        )
    print(f"special_splitted_text\n {splitted_text}")

    # Pretokenize
    # TODO: consider placing the for loop outside the pretokenizer to help 
    # with parallelization.
    pretokenized_text = pretokenize(splitted_text)
    print(f"Pretokenized text\n {pretokenized_text}")

    print("TODO: FIXME Transform to Bytes")

    #bytes = 

    return
          

def split_on_special_tokens(
        input_string: str,
        special_tokens: list[str],
        rejoin: bool = True
        )-> Union[list[str], str]:
    """
        Split a string on special tokens.

        Parameters
        ----------
        input_string : str
            The input string to be split.
        special_tokens : list of str or str
            Special tokens to split the input string on. Each occurrence of a special token will be used as a split point.
        rejoin : bool, optional
            If True, the resulting list of strings will be joined into a single string separated by spaces. 
            If False, returns a list of split strings. Default is True.

        Returns
        -------
        splitted: Union[list[str], str]
            The input string split on the special tokens. If `rejoin` is True, returns a single-element list containing the joined string.
            Note that the split operation (string chunking) will drop all the special_tokens.
        

        Examples
        --------
        >>> split_on_special_tokens("Hello <|endoftext|> world", ["<|endoftext|>"])
        ['Hello  world']

        >>> split_on_special_tokens("foo bar baz", ["bar"], rejoin=False)
        ['foo ', ' baz']

        TODO: Discuss; if we want to rename this to drop_special_tokens and return the joined output?
    """
    
    # sorted_tokens = special_tokens.sort()
    pattern = "|".join(regex.escape(st) for st in special_tokens)
    splitted = regex.split(pattern=pattern, string=input_string)
    if rejoin:
        return " ".join(splitted)

    return splitted


def pretokenize(
        strings: list[str],
        regex_pattern: Optional[str] = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        ) -> list[str]:
    '''

    TODO: add fancy tools like
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    review usage of 'match', 'pattern', 'search' methods of <class '_regex.Scanner'>

    '''
    output = []
    for element in strings:
        # splitted = element.split()
        splitted = regex.finditer(regex_pattern, element)
        output.extend(splitted)

    # output = input_string.split()
    return output

def main():
    train_bpe()

if __name__ == "__main__":
    main()
