import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, dropout):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, 2)
        self.dropout =nn.Dropout(dropout)

    def forward(self, x):
        emb = self.embedding(x) 
        emb=self.dropout(emb)
        emb, (h_n, c_n) = self.rnn(emb)
        output_fw = h_n[-2, :, :]  
        output_bw = h_n[-1, :, :]  
        output = torch.cat([output_fw, output_bw], dim=-1) 
        out = self.linear(output)  
        return F.log_softmax(out, dim=-1)