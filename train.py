import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

def train_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    losses = []
    correct_predictions = 0

    for batch in tqdm(dataloader, desc='Training', leave=False):
        input_ids = batch['input_ids'].to(device)
        labels: Tensor = batch['label'].to(device)
        
        # Prepare target sequences (in this case, labels are sentiment labels 0-6)
        target_input = labels.unsqueeze(1).to(device)  # Reshape labels to 2D tensor
        target_input = target_input.expand(-1, input_ids.size(1)).to(device)  # Expand to match input_ids sequence length
        target_output = target_input  # Target output is the same in this case

        outputs = model(input_ids, target_input).to(device)
        outputs = outputs.view(-1, outputs.shape[-1]).to(device)
        target_output = target_output.reshape(-1).to(device)
        
        loss = loss_fn(outputs, target_output)
        losses.append(loss.item())

        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == target_output)

    return correct_predictions.double() / len(dataloader.dataset), np.mean(losses)

def eval_model(model, dataloader, loss_fn, device):
    model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation', leave=False):
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / len(dataloader.dataset), np.mean(losses)


def get_predictions(model, dataloader, device):
    model.eval()
    predictions = []
    real_values = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Testing', leave=False):
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids)
            _, preds = torch.max(outputs, dim=1)

            predictions.extend(preds)
            real_values.extend(labels)

    predictions = torch.stack(predictions).to(device)
    real_values = torch.stack(real_values).to(device)
    return predictions, real_values
