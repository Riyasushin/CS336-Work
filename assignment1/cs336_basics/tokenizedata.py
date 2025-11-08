from cs336_basics.train_bpe import BPE_Tokenizer
import argparse
import numpy as np
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize text using a trained BPE tokenizer.")
    parser.add_argument("--input", default='/data/CS336-use/TinyStoriesV2-GPT4-valid.txt', help="Path to input text file")
    parser.add_argument("--output", default='/data/CS336-use/Tokens/TinyStories_valid_10000_token_ids.npy', help="Path to save .npy token IDs")
    parser.add_argument("--tokenizer_folder", default='./bpe_on_TinyStories_train', help="Path to trained BPE tokenizer folder (contains vocab.json & merges.txt)")
    parser.add_argument("--special_tokens", nargs="*", default=["<|endoftext|>"], help="List of special tokens")
    parser.add_argument("--verbose", action="store_true", help="Print progress info")
    args = parser.parse_args()

    tokenizer = BPE_Tokenizer.from_file(args.tokenizer_folder, args.special_tokens)
    tokenizer.encode_to_npfile(args.input, args.output)
    
'''
uv run --input /root/autodl-tmp/TinyStoriesV2-GPT4-train.txt --output npy_data/train.npy 
''' 