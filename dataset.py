import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from ast import literal_eval

class SentimentDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_ids = self.inputs[idx]
        input_ids = literal_eval(input_ids)
        label = self.labels[idx]
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

batch_size = int(input('Enter the batch size: '))

print('Loading data...', end=' ')
data = pd.read_csv('data/preprocessed_data.csv')
print('Data loaded')

print('Splitting data...', end=' ')
dataset = SentimentDataset(data['input_ids'], data['label'])

train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
print('Done splitting data')

print()
print(f"Train size: {train_size} ({train_size/len(dataset)*100:.2f}%)")
print(f"Validation size: {val_size} ({val_size/len(dataset)*100:.2f}%)")
print(f"Test size: {test_size} ({test_size/len(dataset)*100:.2f}%)")
print()

print('Creating dataloaders...', end=' ')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, num_workers=2)
print('Dataloaders created')
print()
