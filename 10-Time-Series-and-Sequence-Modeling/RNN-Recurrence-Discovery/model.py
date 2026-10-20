import torch
import torch.nn as nn

from data_prep import RecurrenceData

class RecurrenceModel(nn.Module):
    def __init__(self):
        super(RecurrenceModel, self).__init__()

class RNNPredictor(RecurrenceModel):
    """Q2.2: Deep Learning Predictor (GRU)."""
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(RNNPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)  
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)

class LinearARPredictor(RecurrenceModel):
    """Q2.2: Baseline Predictor (Linear AR)."""
    def __init__(self, history_length):
        super(LinearARPredictor, self).__init__()
        self.linear = nn.Linear(history_length, 1)

    def forward(self, x):
        x_flat = x.squeeze(-1) 
        return self.linear(x_flat).squeeze(-1)
class MLPPredictor(RecurrenceModel):
    def __init__(self, history_length, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(history_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

class CNNPredictor(RecurrenceModel):
    def __init__(self, history_length, channels=16):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        x = x.transpose(1, 2)            # (B, 1, p)
        features = self.cnn(x).squeeze(-1)  
        return self.fc(features).squeeze(-1)

class LSTMPredictor(RecurrenceModel):
    """Q2.2: Deep Learning Predictor (LSTM)."""
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        x shape: (batch, seq_len, input_size)
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # final timestep
        return out.squeeze(-1)
