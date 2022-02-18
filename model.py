import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

from tokenizers import Tokenizer


# TODO: Config file
DIR = os.path.dirname(os.path.realpath(__file__))
DEVICE = "cuda"
PAD_INDEX = 1
EOS_INDEX = 0


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(
            0, max_len, dtype=torch.float).view(-1, 1)  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float(
        ) * (-math.log(10000.0)) / dim_model)  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class Transformer(nn.Module):

    def __init__(
        self,
        tokenizer: Tokenizer,
        num_tokens: int,
        dim_model: int,
        num_heads: int,
        d_hid: int,
        num_layers: int,
        dropout_p: float,
        n_positions: int,
    ):
        super().__init__()

        self.tokenizer = tokenizer

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=n_positions
        )
        self.embedding = nn.Embedding(
            num_tokens, dim_model, padding_idx=PAD_INDEX)

        encoder_layers = TransformerEncoderLayer(
            dim_model,
            num_heads,
            d_hid,
            dropout_p,
            batch_first=False,
            activation=F.gelu,
            norm_first=True,
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, num_layers)

        self.out = nn.Linear(dim_model, num_tokens)

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, src, src_mask=None, src_pad_mask=None) -> torch.tensor:
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.embedding(src) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer_encoder(
            src,
            src_mask,
            src_pad_mask,
        )

        out = self.out(transformer_out)

        return F.log_softmax(out, dim=-1)

    def get_src_mask(self, sz) -> torch.tensor:
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def get_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token).t()

    def predict(self, input_string: str = "<|THREAD|>", max_length: int = 80, temperature: float = 1.0, top_p: float = 1.0) -> str:
        top_p = torch.tensor(top_p, dtype=torch.float, device=DEVICE)
        zero = torch.tensor(0.0, dtype=torch.float, device=DEVICE)
        self.to(DEVICE)
        self.eval()

        input_sequence = self.tokenizer.encode(input_string).ids
        y_input = torch.tensor(
            [input_sequence], dtype=torch.long, device=DEVICE).t()
        self.num_tokens = len(input_sequence)
        init_len = y_input.shape[0]

        for _ in range(max_length - init_len):
            # Get source mask
            src_mask = self.get_src_mask(y_input.size(0)).to(DEVICE)
            pad_mask = self.get_pad_mask(y_input, PAD_INDEX).to(DEVICE)

            pred = self.forward(y_input, src_mask, pad_mask)

            word_weights = pred[-1].squeeze().div(temperature).exp()

            sorted_weights, sorted_weights_indices = torch.sort(
                word_weights, descending=True)
            cumulative_probs = torch.cumsum(sorted_weights, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_weights_indices[sorted_indices_to_remove]
            word_weights[indices_to_remove] = 0 
            word_weights /= top_p

            word_idx = torch.multinomial(word_weights, 1)[0]
            next_item = torch.tensor([[word_idx]], device=DEVICE)

            # Concatenate previous input with predicted best word
            y_input = torch.cat((y_input, next_item), dim=0)

            # Stop if model predicts end of sentence
            if next_item.view(-1).item() == EOS_INDEX:
                break

        return self.tokenizer.decode(y_input.view(-1).tolist(), skip_special_tokens=False)
