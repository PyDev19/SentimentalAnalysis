import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

def train_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    losses = []
    correct_predictions = 0

    for batch in tqdm(dataloader, desc='Training', leave=False):
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

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
