import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        n_inputs,
        hidden_dim=128,
        num_layers=2,
        bidirectional=True,
        dropout=0.2,
        pool_output=True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.pool_output = pool_output
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=n_inputs,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        
        self.n_outputs = hidden_dim * self.num_directions
        self.bn = nn.BatchNorm1d(self.n_outputs)
        
        if pool_output:
            self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.transpose(1, 2)
        
        x_out, _ = self.lstm(x)
        x_out = x_out.transpose(1, 2)
        x_out = self.bn(x_out)
        
        if self.pool_output:
            x_out = self.pool(x_out)
            x_out = x_out.squeeze(-1)
        
        return x_out
