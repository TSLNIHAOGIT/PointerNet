import torch
import torch.nn as nn
# Building your LSTM
# batch_first=True causes input/output tensors to be of shape
# (batch_dim, seq_dim, feature_dim)
# lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first = True, bidirectional = True)



rnn = nn.LSTM(10, 20, 2)
print(rnn)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, (hn, cn) = rnn(input, (h0, c0))