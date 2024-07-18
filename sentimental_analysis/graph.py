import numpy as np
import torch
from torchviz import make_dot
import matplotlib.pyplot as plt

from model import SentimentCNNBiLSTM
from preprocessing import vocab

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load('models/sentiment_model.pth', map_location=device)

dummy_input = torch.zeros((1, 128), dtype=torch.long)
graph = make_dot(model(dummy_input), params=dict(model.named_parameters()))
graph.render('sentiment_model_graph', format='pdf')
graph.render('sentiment_model_graph', format='png')

history = torch.load('models/train_history.pth', map_location=device)
train_losses = history['train_losses']
val_losses = history['val_losses']
train_accs = history['train_accs']
val_accs = history['val_accs']
learning_rates = history['learning_rates']

train_losses = np.array(list(set(train_losses)))
val_losses = np.array(list(set(val_losses)))

train_accs = list(train_accs)
val_accs = list(val_accs)

for i in range(len(train_accs)):
    train_accs[i] = train_accs[i].item()
    val_accs[i] = val_accs[i].item()

train_accs = np.array(train_accs)
val_accs = np.array(val_accs)

epochs = len(train_losses)

plt.figure(figsize=(10, 5))

plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.plot(range(1, epochs + 1), train_accs, label='Train Accuracy')
plt.plot(range(1, epochs + 1), val_accs, label='Validation Accuracy')

plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.title('Metrics')
plt.legend()

plt.savefig('metrics.png')

plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), learning_rates, label='Learning Rate')
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.title('Learning Rate')
plt.legend()
plt.savefig('learning_rate.png')