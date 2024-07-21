import os

import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils

from tqdm import tqdm

# from model import SentimentCNNBiLSTM
from transformer.transformer import Transformer
from dataset import train_dataloader, val_dataloader, test_dataloader
from train import train_epoch, eval_model, get_predictions

use_tpu = input('Use TPU? (y/n): ')
use_tpu = use_tpu.lower() == 'y'

if not use_tpu:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if device == 'cpu':
        torch.set_num_threads(16)
    else:
        torch.backends.cudnn.benchmark = True
else:
    device = xm.xla_device()
    print(f"Using device: {device}")
    
vocab = torch.load(f'models/vocab.pth')

VOCAB_SIZE = len(vocab)
CLASSES = 7
DIMENSIONS = 512
HEADS = 8
LAYERS = 6
HIDDEN_DIMENSIONS = 2048
MAX_SEQ_LEN = 209
DROPOUT = 0.1
LEARNING_RATE = 1e-4
EPOCHS = 10

# model = SentimentCNNBiLSTM(VOCAB_SIZE, EMBEDDING_DIM, CONV_FILTERS, LSTM_HIDDEN_DIM, OUTPUT_DIM, DROPOUT)
# model = model.to(device)

model = Transformer(VOCAB_SIZE, DIMENSIONS, HEADS, LAYERS, HIDDEN_DIMENSIONS, MAX_SEQ_LEN, CLASSES, DROPOUT, device).to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
loss_fn = CrossEntropyLoss(ignore_index=0).to(device)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=0, verbose=True)

train_losses = []
train_accs = []
val_losses = []
val_accs = []
learning_rates = []

for epoch in tqdm(range(EPOCHS), desc='Epochs', leave=True):
    train_acc, train_loss = train_epoch(model, train_dataloader, loss_fn, optimizer, device)
    print(f'Train loss: {train_loss}, Accuracy: {train_acc}')
    train_losses.append(train_loss)

    val_acc, val_loss = eval_model(model, val_dataloader, loss_fn, device)
    print(f'Val loss: {val_loss}, Accuracy: {val_acc}')
    print()

    # scheduler.step(val_loss)

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