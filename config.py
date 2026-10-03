from pathlib import Path
import string

def token_mapper():
    latin_chars = list(string.ascii_lowercase)
    cyrillic_chars = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
    en_vocab_size = len(latin_chars) + 3
    ru_vocab_size = len(cyrillic_chars) + 3
    special_tokens = [' ', '<SOS>', '<EOS>']
    all_tokens = special_tokens + sorted(set(latin_chars + cyrillic_chars))

    token_to_id = {token: idx for idx, token in enumerate(all_tokens)}
    id_to_token = {val: key for key, val in token_to_id.items()}
    return token_to_id, id_to_token, en_vocab_size, ru_vocab_size

def get_config():
    token_to_id, id_to_token, en_vocab_size, ru_vocab_size = token_mapper()
    return {
            "lr" : 5e-4,
            "seq_len" : 13,
            "batch_size" : 32,
            "embedding_dim" : 256,
            "hidden_size" : 512,
            "token_to_id" : token_to_id,
            "id_to_token" : id_to_token,
            "vocab_size" : len(token_to_id),
            "en_vocab_size" : en_vocab_size,
            "ru_vocab_size" : ru_vocab_size,
            }

