import os

import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

from tqdm import tqdm

from transformer.transformer import Transformer
from train import train_epoch, eval_model, get_predictions

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

def train_tpu(rank, flags):
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.core.xla_model as xm
    from dataset import train_dataloader, val_dataloader
    
    device = xm.xla_device()
    
    LEARNING_RATE = 1e-4 * xm.xrt_world_size()
    
    model = Transformer(VOCAB_SIZE, DIMENSIONS, HEADS, LAYERS, HIDDEN_DIMENSIONS, MAX_SEQ_LEN, CLASSES, DROPOUT, device).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
    loss_fn = CrossEntropyLoss(ignore_index=0).to(device)
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    learning_rates = []
    
    epochs = int(input('Enter the number of epochs: '))
    for epoch in range(epochs):
        if xm.is_master_ordinal():
            print(f'Epoch {epoch + 1}/{epochs}')
        
        train_dataloader = pl.ParallelLoader(train_dataloader, [device])
        val_dataloader = pl.ParallelLoader(val_dataloader, [device])
        
        train_acc, train_loss = train_epoch(model, train_dataloader.per_device_loader(device), loss_fn, optimizer, device)
        del train_dataloader
        
        if xm.is_master_ordinal():
            print(f'Train loss: {train_loss}, Accuracy: {train_acc}')
        train_losses.append(train_loss)

        val_acc, val_loss = eval_model(model, val_dataloader.per_device_loader(device), loss_fn, device)
        del val_dataloader
        
        if xm.is_master_ordinal():
            print(f'Val loss: {val_loss}, Accuracy: {val_acc}')
            print()

        # scheduler.step(val_loss)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        learning_rates.append(optimizer.param_groups[0]['lr'])
    
    model.to('cpu')
    optimizer.to('cpu')
    
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
    
def test_tpu(rank, flags):
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.core.xla_model as xm
    from dataset import test_dataloader
    
    device = xm.xla_device()
    
    model = Transformer(VOCAB_SIZE, DIMENSIONS, HEADS, LAYERS, HIDDEN_DIMENSIONS, MAX_SEQ_LEN, CLASSES, DROPOUT, device).to(device)
    loss_fn = CrossEntropyLoss(ignore_index=0).to(device)
    
    test_dataloader = pl.ParallelLoader(test_dataloader, [device])
    accuracy = eval_model(model, test_dataloader.per_device_loader(device), loss_fn, device)
    del test_dataloader
    
    if xm.is_master_ordinal():
        print(f'Test accuracy: {accuracy}')

def train_and_test_tpu(rank, flags):
    train_tpu(rank, flags)
    test_tpu(rank, flags)

def normal_train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if device == 'cpu':
        torch.set_num_threads(16)
    else:
        torch.backends.cudnn.benchmark = True
    
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

        # scheduler.step(val_loss)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        learning_rates.append(optimizer.param_groups[0]['lr'])

if __name__ == '__main__':
    if use_tpu:
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(train_and_test_tpu, nprocs=8, start_method='fork', args=({},))
    else:
        normal_train()
