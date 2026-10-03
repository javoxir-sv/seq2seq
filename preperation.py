import torch
import torch.nn as nn
import polars as pl
import numpy as np
from torch.utils.data import Dataset

class LetterTokenizer:
    def __init__(self, config, dataset_path: str):
        self.dataset_path = dataset_path
        self.config = config
        self.token_to_id = config['token_to_id']
        self.id_to_token = config['id_to_token']
        self.seq_len = config['seq_len']


    def load_data(self) -> pl.DataFrame:
        try:
            return pl.read_csv(self.dataset_path, separator=";", has_header=False, new_columns=["latin", "cyrillic", "gender"])
        except Exception:
            df = pl.read_csv(self.dataset_path)
            first_col = df.columns[0]
            return df.with_columns(
                    pl.col(first_col)
                    .str.split_exact(";", 2)
                    .struct.rename_fields(['latin', 'cyrillic', 'gender'])
                    ).unnest(first_col)


    def _encode_column(self, df: pl.DataFrame, col_name: str) -> list[list[int]]:
        pad_tokens  = [' '] * self.seq_len

        char_list = pl.col(col_name).str.to_lowercase().str.extract_all(r".")

        tokens = pl.concat_list(
                pl.lit(["<SOS>"]),
                char_list,
                pl.lit(["<EOS>"])
                )
        encoded_expr = (
                pl.concat_list([tokens, pl.lit(pad_tokens)])
                .list.slice(0, self.seq_len)
                .list.eval(pl.element().replace_strict(self.token_to_id, default=self.token_to_id.get(' ', 0)))
                )
        return df.select(encoded_expr).to_series().to_list()


    def encode(self):
        df = self.load_data()
        latin_tokenized = self._encode_column(df, 'latin')
        cyrillic_tokenized = self._encode_column(df, 'cyrillic')
        gender = df['gender'].to_list()
        return latin_tokenized, cyrillic_tokenized, gender



class CustomDataset(Dataset):
    def __init__(self, x, y, transform=None, target_transform=None):
        self.x = torch.tensor(x, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y


class PE_sinusoidal(nn.Module):
    def __init__(self, d_model:int, seq_len:int, grad:bool=False):
        super().__init__()
        self.d_model = d_model # d_model is the embedding dimension, the embeddings and the positional values has to match, if 256 embeds then 256 positional values to add
        self.seq_len = seq_len
        self.grad = grad # if we want to train the weights or not

        pe = torch.zeros(seq_len, d_model) # (seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)) # it's a scalar

        pe[:, 0::2] = torch.sin(position * div_term) # the same dimension, but only applied to even numbers: 0::2
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        if grad:
            self.pe = nn.Parameter(pe)
        else:
            self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.shape[1], :]
        


