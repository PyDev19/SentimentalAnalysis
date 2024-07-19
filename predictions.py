import torch

from preprocessing import clean_text, tokenize, encode_tokens, pad_sequence
from model import SentimentCNNBiLSTM

device = 'cuda' if torch.cuda.is_available() else 'cpu'

vocab = torch.load('models/vocab.pth', map_location=torch.device(device))
label_encoder = torch.load('models/label_encoder.pth', map_location=torch.device(device))

model = torch.load('models/sentiment_model.pth', map_location=torch.device(device))

while True:
    prompt = input("Enter a sentence: ")
    if prompt == 'exit':
        break
    else:
        prompt = clean_text(prompt)
        prompt = tokenize(prompt)
        prompt = encode_tokens(prompt, vocab)
        prompt = pad_sequence(prompt, 128)
        tensor = torch.tensor(prompt, dtype=torch.long).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            output = model(tensor)

        probabilities = torch.softmax(output, dim=1)

        _, predicted_class = torch.max(output, dim=1)
        predicted_sentiment = predicted_class.item()

        predicted_sentiment_label = label_encoder.classes_[predicted_sentiment] - 1

        print(f"Predicted Sentiment: {predicted_sentiment+1}")