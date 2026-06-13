import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dropout=0.5):
        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual

        out = self.relu(out)
        return out

class EmbeddingCRNN(nn.Module):
    def __init__(self, num_classes=6, embedding_dim=32, cnn_channels=64, lstm_hidden=128):
        super(EmbeddingCRNN, self).__init__()

        self.pitch_embedding = nn.Embedding(num_embeddings=128, embedding_dim=embedding_dim)

        cnn_input_size = embedding_dim + 3

        self.input_conv = nn.Conv1d(cnn_input_size, cnn_channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm1d(cnn_channels)
        self.relu = nn.ReLU()

        self.res1 = ResidualBlock(cnn_channels, kernel_size=3, dropout=0.5)
        self.res2 = ResidualBlock(cnn_channels, kernel_size=3, dropout=0.5)

        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.4
        )

        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x):
        pitch_idx, rest_features = x

        emb = self.pitch_embedding(pitch_idx)

        x = torch.cat([emb, rest_features], dim=2)

        x = x.permute(0, 2, 1)

        out = self.input_conv(x)
        out = self.bn_in(out)
        out = self.relu(out)
        out = self.res1(out)
        out = self.res2(out)

        out = out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(out)

        logits = self.fc(lstm_out)

        return logits.permute(0, 2, 1)
    
