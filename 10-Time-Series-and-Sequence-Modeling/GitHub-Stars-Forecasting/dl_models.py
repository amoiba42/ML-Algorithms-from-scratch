import torch
import torch.nn as nn

class RNNForecaster(nn.Module):
    """
    Q3.2.2: Recurrent Neural Network (GRU) for sequence forecasting.
    """
    def __init__(self, lookback, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(RNNForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Use GRU for efficiency and performance
        self.rnn = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True
        )
        # Output layer maps the last hidden state to the single prediction
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, lookback, 1)
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Forward pass GRU
        out, _ = self.rnn(x, h0)  
        # Pass the hidden state of the LAST time step to the linear layer
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1) # Output shape: (batch_size,)


class CNNForecaster(nn.Module):
    """
    Q3.2.2: 1D Convolutional Neural Network for sequence forecasting.
    """
    def __init__(self, lookback, input_size=1, filters=32, kernel_size=3):
        super(CNNForecaster, self).__init__()
        # Input: (Batch, 1 channel, lookback length) -> transpose sequence dim for conv1d
        self.conv1 = nn.Conv1d(
            in_channels=input_size, 
            out_channels=filters, 
            kernel_size=kernel_size,
            padding='same'
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        # Calculate the size after pooling (Approx lookback / 2)
        pooled_length = lookback // 2 
        self.fc1 = nn.Linear(filters * pooled_length, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # Input shape: (Batch, Lookback, Input_size=1)
        
        # Permute to (Batch, Input_size=1, Lookback) for Conv1D
        x = x.permute(0, 2, 1) 
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        # Flatten the output for the fully connected layers
        x = x.contiguous().view(x.size(0), -1) 
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x.squeeze(-1) # Output shape: (batch_size,)