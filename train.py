import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Tuple, List
from torch import nn, optim, Tensor

def train_epoch(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    import torch_xla.core.xla_model as xm
    
    model.train()
    losses: List[float] = []
    correct_predictions: int = 0

    for batch in tqdm(dataloader, desc='Training', leave=False):        
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad(set_to_none=True)

        outputs = model(input_ids)
        outputs = outputs.contiguous().view(-1, 7)
        labels = labels.contiguous().view(-1)
        
        loss = loss_fn(outputs, labels)
        losses.append(loss.item())

        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        # optimizer.step()
        xm.optimizer_step(optimizer)

        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)

    accuracy = correct_predictions.double() / len(dataloader.dataset)
    average_loss = np.mean(losses)
    return accuracy, average_loss

def eval_model(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    losses: List[float] = []
    correct_predictions: int = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation', leave=False):
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids)
            outputs = outputs.contiguous()
            outputs = outputs.view(-1, 7)
            labels = labels.contiguous().view(-1)
            
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
            
            outputs = model(input_ids)
            
            _, preds = torch.softmax(outputs, dim=1)

            predictions.extend(preds)
            real_values.extend(labels)

    predictions = torch.stack(predictions).to(device)
    real_values = torch.stack(real_values).to(device)
    return predictions, real_values
