from transformers import AutoTokenizer


def get_tokenizer(args, verbose=True):
    """GPT-2 tokenizer at the pinned --tokenizer-revision."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2", revision=getattr(args, "tokenizer_revision", "main"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set model max length
    max_length = getattr(args, 'max_length', None) or getattr(args, 'sequence_length', 1024)
    tokenizer.model_max_length = max_length

    if verbose:
        print("Using tokenizer: gpt2")
        print(f"Vocabulary size: {len(tokenizer)}")
        print(f"Max length: {max_length}")
        print(f"Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    return tokenizer
