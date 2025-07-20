import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import requests
import nltk
from nltk.tokenize import word_tokenize
import random

# ======== Download NLTK data (run once) ========
nltk.download('punkt')

# =============== Sample Dataset (Word-Level) ================
shakespeare_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
sample_text = requests.get(shakespeare_url).text
sample_text = sample_text[:100000]  # Use a subset for fast testing (increase for better results)

# ----------------- Tokenizer -----------------
class WordTokenizer:
    def __init__(self, text):
        # Tokenize and build vocab
        tokens = word_tokenize(text)
        self.vocab = sorted(set(tokens))
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.tokens = tokens

    def encode(self, words):
        # Accepts list of words or single string (splits string)
        if isinstance(words, str):
            words = word_tokenize(words)
        return [self.word2idx[w] for w in words if w in self.word2idx]

    def decode(self, idxs):
        return ' '.join([self.idx2word[i] for i in idxs])

tokenizer = WordTokenizer(sample_text)

# --------------- Custom Dataset -----------------
class WordDataset(Dataset):
    def __init__(self, tokens, tokenizer, seq_len=10):
        self.data = tokenizer.encode(tokens)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+1:idx+self.seq_len+1]
        return torch.tensor(x), torch.tensor(y)

# ============== LSTM RNN Model Definition =================
class LSTMRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=64, hidden_size=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)
    def forward(self, x, h=None):
        x = self.embed(x)
        out, h = self.rnn(x, h)
        logits = self.fc(out)
        return logits, h

# ========== Training & Fine-Tuning Function ==============
def train_model(model, dataset, epochs=5, lr=0.01):
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in dataloader:
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")

# ========== Test Case: Model Predicts Next Word ==========
def test_prediction(model, seed, tokenizer, steps=20, temperature=1.0):
    model.eval()
    input_idxs = tokenizer.encode(seed)
    if len(input_idxs) == 0:
        print(f"Seed not in vocab: {seed}")
        return
    input_seq = torch.tensor([input_idxs])
    h = None
    output = word_tokenize(seed)
    with torch.no_grad():
        for _ in range(steps):
            logits, h = model(input_seq, h)
            last_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(last_logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()
            next_word = tokenizer.idx2word[next_idx]
            output.append(next_word)
            input_seq = torch.tensor([[next_idx]])
    print(f"Test Generation: {' '.join(output)}")

# ========== Main Script ==========
if __name__ == "__main__":
    print("Tokenizing and preparing data...")
    dataset = WordDataset(tokenizer.tokens, tokenizer, seq_len=10)
    vocab_size = len(tokenizer.vocab)
    print(f"Vocab size: {vocab_size}, Dataset size: {len(dataset)}")
    model = LSTMRNN(vocab_size)
    print("Training LSTM on sample text...")
    train_model(model, dataset, epochs=5)

    print("\nTesting model prediction from seeds:")
    test_prediction(model, "ROMEO :", tokenizer)
    test_prediction(model, "To be or not", tokenizer)
    test_prediction(model, "JULIET :", tokenizer)
    test_prediction(model, "ACT II . SCENE I .", tokenizer)