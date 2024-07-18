import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import SentimentCNNBiLSTM
from preprocessing import vocab
from dataset import train_dataloader, val_dataloader, test_dataloader
from train import train_epoch, eval_model, get_predictions

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

if device == 'cpu':
    torch.set_num_threads(16)
else:
    torch.backends.cudnn.benchmark = True

VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 100
CONV_FILTERS = 128
LSTM_HIDDEN_DIM = 128
OUTPUT_DIM = 6
DROPOUT = 0.5
LEARNING_RATE = 1e-4
EPOCHS = 20

model = SentimentCNNBiLSTM(VOCAB_SIZE, EMBEDDING_DIM, CONV_FILTERS, LSTM_HIDDEN_DIM, OUTPUT_DIM, DROPOUT)
model = model.to(device)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = CrossEntropyLoss().to(device)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=0, verbose=True)

train_losses = []
train_accs = []
val_losses = []
val_accs = []
learning_rates = []

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(model, train_dataloader, loss_fn, optimizer, device)
    print(f'Train loss: {train_loss}, Accuracy: {train_acc}')
    train_losses.append(train_loss)

    val_acc, val_loss = eval_model(model, val_dataloader, loss_fn, device)
    print(f'Val loss: {val_loss}, Accuracy: {val_acc}')
    print()

    scheduler.step(val_loss)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    learning_rates.append(optimizer.param_groups[0]['lr'])

predictions, true_labels = get_predictions(model, test_dataloader, device)
accuracy = (predictions == true_labels).sum() / len(true_labels)
print(f'Accuracy: {accuracy}')

import os
if not os.path.exists('models'):
    os.makedirs('models')

torch.save(model, 'models/sentiment_model.pth')
torch.save(optimizer, 'models/sentiment_optimizer.pth')
torch.save({
    'train_losses': train_losses,
    'train_accs': train_accs,
    'val_losses': val_losses,
    'val_accs': val_accs,
    'learning_rates': learning_rates
}, 'models/train_history.pth')