import math
import os
import numpy as np
import pandas as pd
import torch
import json
import jieba
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, dataframe, char_to_id, max_length=MAX_LEN):
        self.data = dataframe
        self.char_to_id = char_to_id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence1 = self.data.iloc[index]['sentence1']
        sentence2 = self.data.iloc[index]['sentence2']
        if 'label' in self.data.columns:
            label = self.data.iloc[index]['label']
        label = int(label)

        combined_tokens = ['[CLS]'] + [char for char in sentence1] + ['[SEP]'] + [char for char in sentence2] + ['[SEP]']
        segment_ids = [0] * (len(sentence1) + 2) + [1] * (len(sentence2) + 1)

        combined_ids = [self.char_to_id.get(char, self.char_to_id['[MASK]']) for char in combined_tokens]
        combined_ids = torch.nn.functional.pad(torch.tensor(combined_ids), (0, self.max_length - len(combined_ids)))
        segment_ids = torch.nn.functional.pad(torch.tensor(segment_ids), (0, self.max_length - len(segment_ids)))
        # label = torch.tensor(label)

        return combined_ids, segment_ids, label

train_dataset = TextDataset(train_df, char_to_id)
dev_dataset = TextDataset(dev_df, char_to_id)
test_dataset = TextDataset(test_df, char_to_id)

def collate_fn(batch_data, pad_val=0, max_seq_len=MAX_LEN):
    input_ids, segment_ids, labels = [], [], []
    max_len = 0
    for example in batch_data:
        input_id, segment_id, label = example
        # cut
        input_ids.append(input_id[:max_seq_len])
        segment_ids.append(segment_id[:max_seq_len])
        labels.append(label)
        # max
        max_len = max(max_len, len(input_id))
    # pad
    for i in range(len(labels)):
        input_ids[i] = torch.cat([input_ids[i], torch.tensor([pad_val] * (max_len - len(input_ids[i])))])
        segment_ids[i] = torch.cat([segment_ids[i], torch.tensor([pad_val] * (max_len - len(segment_ids[i])))])

    return torch.stack(input_ids), torch.stack(segment_ids), torch.tensor(labels)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# mini-batch
print("mini-batch sample:")
for idx, item in enumerate(train_dataloader):
    if idx == 0:
        print(item)
        break

class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size, padding_idx=0):
        super(WordEmbedding, self).__init__()
        self.emb_size = emb_size
        self.word_embedding = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        nn.init.normal_(self.word_embedding.weight, mean=0.0, std=emb_size ** -0.5)

    def forward(self, word):
        word_emb = (self.emb_size ** 0.5) * self.word_embedding(word)
        return word_emb

class SegmentEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(SegmentEmbedding, self).__init__()
        self.emb_size = emb_size
        self.seg_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size)

    def forward(self, word):
        seg_embedding = self.seg_embedding(word)
        return seg_embedding
    
def get_sinusoid_encoding(position_size, hidden_size):
    def cal_angle(pos, hidden_idx):
        return pos / np.power(10000, 2 * (hidden_idx // 2) / hidden_size)

    def get_posi_angle_vec(pos):
        return [cal_angle(pos, hidden_j) for hidden_j in range(hidden_size)]

    sinusoid = np.array([get_posi_angle_vec(pos_i) for pos_i in range(position_size)])
    sinusoid[:, 0::2] = np.sin(sinusoid[:, 0::2])
    sinusoid[:, 1::2] = np.cos(sinusoid[:, 1::2])
    return torch.tensor(sinusoid, dtype=torch.float32)

class PositionalEmbedding(nn.Module):
    def __init__(self, max_length, emb_size):
        super(PositionalEmbedding, self).__init__()
        self.emb_size = emb_size
        self.max_length = max_length
        self.register_buffer('pos_encoder', get_sinusoid_encoding(max_length, self.emb_size))
    
    def forward(self, pos):
        # Ensure that pos is within valid range
        pos = torch.clamp(pos, 0, self.max_length - 1)
        pos_emb = self.pos_encoder[pos]
        pos_emb = pos_emb.detach()  # 关闭位置编码的梯度更新
        return pos_emb
    
class TransformerEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, hidden_dropout_prob, position_size, segment_size):
        super(TransformerEmbeddings, self).__init__()
        self.word_embeddings = WordEmbedding(vocab_size, hidden_size)
        self.position_embeddings = PositionalEmbedding(position_size, hidden_size)
        self.segment_embeddings = SegmentEmbedding(segment_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, segment_ids=None, position_ids=None):
        if position_ids is None:
            ones = torch.ones_like(input_ids, dtype=torch.long)
            seq_length = torch.cumsum(ones, dim=-1)
            position_ids = seq_length - ones
            position_ids = position_ids.clamp(0, self.position_embeddings.max_length - 1)

        input_embeddings = self.word_embeddings(input_ids)
        segment_embeddings = self.segment_embeddings(segment_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_embeddings + segment_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        embeddings = embeddings.to(torch.float32)
        return embeddings
    
# Feed Forward
# Actually, it's a two-layer MLP
class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, pw_num_outputs, **kwargs) -> None:
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, pw_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
    
class AddNorm(nn.Module): 
    """The residual connection followed by layer normalization."""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
    
def transpose_qkv(X, num_heads):

    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    X.shape = (batch_size, num_steps, num_hiddens)
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

class multiheadAttention (nn.Module):
    def __init__(self, query_size, key_size, value_size,
                 num_hiddens, num_heads, dropout, bias=False, 
                 *args, **kwargs) -> None:
        super(multiheadAttention, self).__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        """Note
        concat all the heads together for matrix multiplication
        """
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
        output = self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

class EncoderBlock(nn.Module):
    def __init__(self, query_size, key_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout, bias=False, **kwargs) -> None:
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = multiheadAttention(query_size, key_size, value_size, num_hiddens, num_heads, dropout, bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        attention_output, attention_weights = self.attention(X, X, X, valid_lens) 
        Y = self.addnorm1(X, attention_output)  # self-attention
        return self.addnorm2(Y, self.ffn(Y))
    
"""
- Self-Attention
- Add & Norm
- Feed Forward
- Add & Norm
- Encoder Block
"""
class TransformerEncoder(d2l.Encoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = TransformerEmbeddings(vocab_size, num_hiddens, dropout, position_size=MAX_LEN, segment_size=2)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            # print(X.shape)
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
            print(self.attention_weights[i].shape)
        return X
    
