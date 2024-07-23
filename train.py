import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Tuple, List
from torch import nn, optim, Tensor
from torch.cuda.amp import autocast

def train_epoch(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, optimizer: optim.Optimizer, device: torch.device, scaler, scheduler) -> Tuple[float, float]:    
    model.train()
    losses: List[float] = []
    correct_predictions: int = 0

    for batch in tqdm(dataloader, desc='Training', leave=True, position=0):        
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            outputs = model(input_ids, mask=None)
            loss = loss_fn(outputs, labels)
        
        losses.append(loss.item())

        scaler.scale(loss).backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)
    
    scheduler.step()

    accuracy = correct_predictions.double() / len(dataloader.dataset)
    average_loss = np.mean(losses)
    return accuracy, average_loss

def eval_model(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    losses: List[float] = []
    correct_predictions: int = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation', leave=True, position=0):
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            with autocast():  # Enable mixed precision during evaluation
                outputs = model(input_ids, mask=None)
                loss = loss_fn(outputs, labels)
                        
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    accuracy = correct_predictions.double() / len(dataloader.dataset)
    average_loss = np.mean(losses)
    return accuracy, average_loss


def get_predictions(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[Tensor, Tensor]:
    model.eval()
    predictions: List[Tensor] = []
    real_values: List[Tensor] = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Testing', leave=False):
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            with autocast():
                outputs = model(input_ids, mask=None)
                _, preds = torch.max(outputs, dim=1)
            
            _, preds = torch.softmax(outputs, dim=1)

            predictions.extend(preds)
            real_values.extend(labels)

    predictions = torch.stack(predictions).to(device)
    real_values = torch.stack(real_values).to(device)
    return predictions, real_values
