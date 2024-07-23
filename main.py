import os

import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformer.transformer import Transformer
from train import train_epoch, eval_model, get_predictions
from dataset import train_dataloader, val_dataloader, test_dataloader

use_tpu = True
    
vocab = torch.load(f'models/vocab.pth')
print(f'Vocab size: {len(vocab)}')

VOCAB_SIZE = len(vocab)
EMBED_SIZE = 256
NUM_LAYERS = 6
HEADS = 8
FORWARD_EXPANSION = 4
DROPOUT = 0.1
MAX_LENGTH = 512
NUM_CLASSES = 7
LEARNING_RATE = 1e-6

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if device == 'cpu':
        torch.set_num_threads(16)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    model = Transformer(EMBED_SIZE, NUM_LAYERS, HEADS, device, FORWARD_EXPANSION, DROPOUT, MAX_LENGTH, VOCAB_SIZE, NUM_CLASSES).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
    loss_fn = CrossEntropyLoss(ignore_index=0).to(device)
    scaler = GradScaler()
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    learning_rates = []
    
    epochs = int(input('Enter the number of epochs: '))
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)
        
        train_acc, train_loss = train_epoch(model, train_dataloader, loss_fn, optimizer, device, scaler, scheduler)
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
    
    torch.save(model.state_dict(), 'models/model_state.pth')
    torch.save(model, 'models/model.pth')
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
    
    model = Transformer(EMBED_SIZE, NUM_LAYERS, HEADS, device, FORWARD_EXPANSION, DROPOUT, MAX_LENGTH, VOCAB_SIZE, NUM_CLASSES).to(device)
    model.load_state_dict(torch.load('models/model.pth'))
    loss_fn = CrossEntropyLoss(ignore_index=0).to(device)
        
    test_acc, test_loss = get_predictions(model, test_dataloader, loss_fn, device)
    print(f'Test loss: {test_loss}, Accuracy: {test_acc}')

if __name__ == '__main__':
    train()
    test()
