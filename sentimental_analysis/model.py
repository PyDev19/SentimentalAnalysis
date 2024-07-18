import torch
from torch import nn
from torch.nn.functional import relu 
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from sklearn.base import BaseEstimator, ClassifierMixin
from preprocessing import vocab
from dataset import train_dataset, val_dataset, test_dataset
from train import train_epoch, eval_model, get_predictions

class Attention(nn.Module):
    def __init__(self, lstm_hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(lstm_hidden_dim * 2, 1)

    def forward(self, lstm_out):
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended = torch.sum(lstm_out * attention_weights, dim=1)
        return attended

class SentimentCNNBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, conv_filters, lstm_hidden_dim, output_dim, dropout):
        super(SentimentCNNBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=vocab['<PAD>'])

        self.conv1d_1 = nn.Conv1d(in_channels=embedding_dim, out_channels=conv_filters, kernel_size=3, padding=2)
        self.conv1d_2 = nn.Conv1d(in_channels=conv_filters, out_channels=conv_filters, kernel_size=5, padding=2)
        self.conv1d_3 = nn.Conv1d(in_channels=conv_filters, out_channels=conv_filters, kernel_size=7, padding=2)
        
        self.maxpool = nn.MaxPool1d(kernel_size=2)

        self.bilstm = nn.LSTM(conv_filters, lstm_hidden_dim, num_layers=2, bidirectional=True, batch_first=True)
        self.attention = Attention(lstm_hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(lstm_hidden_dim * 2, output_dim)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        embedded = embedded.permute(0, 2, 1)

        conv_out = relu(self.conv1d_1(embedded))
        conv_out = relu(self.conv1d_2(conv_out))
        conv_out = relu(self.conv1d_3(conv_out))
        
        pooled_out = self.maxpool(conv_out).permute(0, 2, 1)

        lstm_out, _ = self.bilstm(pooled_out)
        attended = self.attention(lstm_out)
        attended = self.dropout(attended)

        output = self.fc(attended)
        return output


