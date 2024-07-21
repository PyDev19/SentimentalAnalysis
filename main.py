import os

import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

from tqdm import tqdm

from transformer.transformer import Transformer
from train import train_epoch, eval_model, get_predictions
from dataset import train_dataloader, val_dataloader, test_dataloader

use_tpu = True
    
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

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if device == 'cpu':
        torch.set_num_threads(16)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    model = Transformer(VOCAB_SIZE, DIMENSIONS, HEADS, LAYERS, HIDDEN_DIMENSIONS, MAX_SEQ_LEN, CLASSES, DROPOUT, device).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
    loss_fn = CrossEntropyLoss(ignore_index=0).to(device)
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    learning_rates = []
    
    epochs = int(input('Enter the number of epochs: '))
    for epoch in tqdm(range(epochs), desc='Epochs', leave=True):
        train_acc, train_loss = train_epoch(model, train_dataloader, loss_fn, optimizer, device)
        print(f'Train loss: {train_loss}, Accuracy: {train_acc}')
        train_losses.append(train_loss)

        val_acc, val_loss = eval_model(model, val_dataloader, loss_fn, device)
        print(f'Val loss: {val_loss}, Accuracy: {val_acc}')
        print()

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        learning_rates.append(optimizer.param_groups[0]['lr'])
    
    history = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'learning_rates': learning_rates
    }
    
    if not os.path.exists('models'):
        os.makedirs('models')
    
    torch.save(model.state_dict(), 'models/model.pth')
    torch.save(optimizer.state_dict(), 'models/optimizer.pth')
    torch.save(history, 'models/history.pth')

def test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if device == 'cpu':
        torch.set_num_threads(16)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    model = Transformer(VOCAB_SIZE, DIMENSIONS, HEADS, LAYERS, HIDDEN_DIMENSIONS, MAX_SEQ_LEN, CLASSES, DROPOUT, device).to(device)
    model.load_state_dict(torch.load('models/model.pth'))
    loss_fn = CrossEntropyLoss(ignore_index=0).to(device)
        
    test_acc, test_loss = get_predictions(model, test_dataloader, loss_fn, device)
    print(f'Test loss: {test_loss}, Accuracy: {test_acc}')

if __name__ == '__main__':
    train()
    test()
