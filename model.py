"""Character unigram/bigram TF-IDF -> small PyTorch MLP."""
import math
import re
from collections import Counter
import torch
from torch import nn

def tokens(text):
    chars = re.sub(r'\s+', '', text.lower())
    return list(chars) + [chars[i:i+2] for i in range(len(chars)-1)]

def fit_vocab(texts):
    counts = Counter(t for text in texts for t in set(tokens(text)))
    vocab = {t: i for i, t in enumerate(sorted(counts))}
    idf = [math.log((1 + len(texts)) / (1 + counts[t])) + 1 for t in vocab]
    return vocab, idf

def encode(texts, vocab, idf):
    x = torch.zeros(len(texts), len(vocab))
    for row, text in enumerate(texts):
        for token, count in Counter(tokens(text)).items():
            if token in vocab:
                i = vocab[token]
                x[row, i] = count * idf[i]
    return torch.nn.functional.normalize(x, p=2, dim=1)

class IntentNet(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(size, 48), nn.ReLU(), nn.Dropout(.15), nn.Linear(48, 6))
    def forward(self, x):
        return self.layers(x)

def load_model(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = IntentNet(len(checkpoint['vocab']))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint['vocab'], checkpoint['idf']
