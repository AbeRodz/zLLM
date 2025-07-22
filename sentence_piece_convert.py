#!/usr/bin/env python3

import json
import sys
from pathlib import Path
from sentencepiece import SentencePieceProcessor


class SentencePieceTokenTypes:
    NORMAL = 0
    UNKNOWN = 1
    CONTROL = 2
    UNUSED = 3
    BYTE = 4
    USER_DEFINED = 5


def does_token_look_special(token: str) -> bool:
    return (
        token.startswith("<") and token.endswith(">")
        or token.startswith("[") and token.endswith("]")
    )


def load_tokenizer(model_dir: Path):
    tokenizer_path = model_dir / "tokenizer.model"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"tokenizer.model not found at {tokenizer_path}")

    tokenizer = SentencePieceProcessor()
    tokenizer.LoadFromFile(str(tokenizer_path))

    vocab_size = tokenizer.vocab_size()

    tokens = {}
    for token_id in range(vocab_size):
        piece = tokenizer.IdToPiece(token_id)
        score = tokenizer.GetScore(token_id)

        entry = {
            "piece": piece,
            "score": score,
            "is_unknown": tokenizer.IsUnknown(token_id),
            "is_control": tokenizer.IsControl(token_id),
            "is_unused": tokenizer.IsUnused(token_id),
            "is_byte": tokenizer.IsByte(token_id),
            "type": SentencePieceTokenTypes.NORMAL,  # Default, corrected below
        }

        if entry["is_unknown"]:
            entry["type"] = SentencePieceTokenTypes.UNKNOWN
        elif entry["is_control"]:
            entry["type"] = SentencePieceTokenTypes.CONTROL
        elif entry["is_unused"]:
            entry["type"] = SentencePieceTokenTypes.UNUSED
        elif entry["is_byte"]:
            entry["type"] = SentencePieceTokenTypes.BYTE

        tokens[str(token_id)] = entry

    return tokens, vocab_size


def apply_added_tokens(tokens, vocab_size, model_dir: Path):
    added_tokens_file = model_dir / "added_tokens.json"
    if not added_tokens_file.is_file():
        return

    with added_tokens_file.open("r", encoding="utf-8") as f:
        added_tokens = json.load(f)

    for key, token_id in added_tokens.items():
        if token_id >= vocab_size:
            print(f"⚠️ Ignore token {token_id}: out of range")
            continue

        tokens[str(token_id)] = {
            "piece": key,
            "score": -1000.0,
            "is_unknown": False,
            "is_control": False,
            "is_unused": False,
            "is_byte": False,
            "type": SentencePieceTokenTypes.USER_DEFINED,
        }


def apply_tokenizer_config(tokens, vocab_size, model_dir: Path):
    config_file = model_dir / "tokenizer_config.json"
    if not config_file.is_file():
        return

    with config_file.open("r", encoding="utf-8") as f:
        config = json.load(f)

    decoder = config.get("added_tokens_decoder", {})
    for token_id_str, token_data in decoder.items():
        token_id = int(token_id_str)
        if token_id >= vocab_size:
            print(f"⚠️ Ignore token {token_id}: out of range")
            continue

        token = token_data["content"]

        current = tokens.get(token_id_str, None)
        if current and current["piece"] != token:
            print(
                f"⚠️ Replacing token {token_id}: {current['piece']} -> {token}"
            )

        is_control = token_data.get("special", False) or does_token_look_special(token)

        if is_control:
            token_type = SentencePieceTokenTypes.CONTROL
        else:
            token = token.replace("\u2581", " ")  # Replace ▁ with space
            token_type = SentencePieceTokenTypes.USER_DEFINED

        tokens[token_id_str] = {
            "piece": token,
            "score": -1000.0,
            "is_unknown": False,
            "is_control": is_control,
            "is_unused": False,
            "is_byte": False,
            "type": token_type,
        }


def pad_tokens_if_needed(tokens, vocab_size):
    current_size = len(tokens)
    if current_size >= vocab_size:
        return

    pad_count = vocab_size - current_size
    print(f"Padding vocab with {pad_count} tokens")

    next_id = current_size
    for i in range(1, pad_count + 1):
        tokens[str(next_id)] = {
            "piece": f"[PAD{i}]",
            "score": -1000.0,
            "is_unknown": False,
            "is_control": False,
            "is_unused": True,
            "is_byte": False,
            "type": SentencePieceTokenTypes.UNUSED,
        }
        next_id += 1

def extract_special_token_ids(
    tokens: dict[str, dict],
    model_dir: Path,
    vocab_size: int,
    special_token_types: list[str] = ["unk", "pad", "bos", "eos", "cls", "sep", "mask"]
) -> dict[str, int]:
    special_token_ids = {}

    def find_token_id_by_piece(piece: str) -> int | None:
        for id_str, tok in tokens.items():
            if tok["piece"] == piece:
                return int(id_str)
        return None

    # Search tokenizer_config.json
    config_files = ["tokenizer_config.json", "config.json"]
    for name in config_files:
        path = model_dir / name
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        for typ in special_token_types:
            # If the id is defined, take it
            id_key = f"{typ}_token_id"
            token_id = config.get(id_key)
            if isinstance(token_id, int) and token_id < vocab_size:
                special_token_ids[typ] = token_id
                continue

            # Try to find from token string
            token_key = f"{typ}_token"
            token_value = config.get(token_key)
            if isinstance(token_value, dict):
                token_value = token_value.get("content")
            if isinstance(token_value, str):
                found_id = find_token_id_by_piece(token_value)
                if found_id is not None and found_id < vocab_size:
                    special_token_ids[typ] = found_id

    return special_token_ids

def load_add_special_token_flags(model_dir: Path) -> dict[str, bool]:
    tokenizer_config_file = model_dir / "tokenizer_config.json"
    flags = {}
    if not tokenizer_config_file.is_file():
        return flags
    with tokenizer_config_file.open("r", encoding="utf-8") as f:
        config = json.load(f)
    for typ in ["bos", "eos", "unk", "pad", "cls", "sep", "mask"]:
        val = config.get(f"add_{typ}_token")
        if isinstance(val, bool):
            flags[typ] = val
    return flags


def export_tokenizer_json(tokens, out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(tokens, f, indent=2, ensure_ascii=False)
    print(f"✔️ Exported tokenizer to {out_path}")

def extract_chat_template(model_dir: Path) -> str | None:
    config_files = ["tokenizer_config.json", "config.json"]
    for name in config_files:
        path = model_dir / name
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        template = config.get("chat_template")
        if isinstance(template, str):
            return template
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python sentencepiece_export.py <model_dir>")
        sys.exit(1)

    model_dir = Path(sys.argv[1])
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {model_dir}")

    tokens, vocab_size = load_tokenizer(model_dir)
    apply_added_tokens(tokens, vocab_size, model_dir)
    apply_tokenizer_config(tokens, vocab_size, model_dir)
    pad_tokens_if_needed(tokens, vocab_size)
    special_token_ids = extract_special_token_ids(tokens, model_dir, vocab_size)
    add_special_tokens = load_add_special_token_flags(model_dir)
    chat_template = extract_chat_template(model_dir)

    export = {
        "tokens": tokens,
        "special_tokens": special_token_ids,
        "add_special_tokens": add_special_tokens,
    }
    if chat_template is not None:
        export["chat_template"] = chat_template


    export_tokenizer_json(export, model_dir / "tokenizer_export.json")



if __name__ == "__main__":
    main()
