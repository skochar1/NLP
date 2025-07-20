import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import requests


# =============== Sample Dataset (Character-Level) ================
shakespeare_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
sample_text = requests.get(shakespeare_url).text
sample_text = sample_text[:10000] # comment out when have access to GPU/more CPU

# ----------------- Tokenizer -----------------
class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}
    
    def encode(self, s):
        return [self.char2idx[ch] for ch in s]
    
    def decode(self, arr):
        return ''.join([self.idx2char[i] for i in arr])
    
tokenizer = CharTokenizer(sample_text)

# --------------- Custom Dataset -----------------
class CharDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len=50):
        self.data = tokenizer.encode(text)
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+1:idx+self.seq_len+1]
        return torch.tensor(x), torch.tensor(y)

# ============== RNN Model Definition =================
class RNN(nn.Module):
    def __init__(self, vocab_size, embed_size=32, hidden_size=64):
        super(RNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, h=None):
        x = self.embed(x)
        out, h = self.rnn(x, h)
        logits = self.fc(out)
        return logits, h
    
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
def train_model(model, dataset, epochs=10, lr=0.01, freeze_layers=False, fine_tune_last=False):
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Fine-tuning: Freeze all but the last layer (if requested)
    if freeze_layers:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    
    # Or, fine-tune only the last layer (as an example)
    if fine_tune_last:
        for name, param in model.named_parameters():
            if 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in dataloader:
            optimizer.zero_grad()
            logits, _ = model(x)
            # Reshape for loss calculation
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")

# ========== Error Handling Example ============
def try_training():
    try:
        vocab_size = len(tokenizer.chars)
        model = LSTMRNN(vocab_size)
        dataset = CharDataset(sample_text, tokenizer)
        train_model(model, dataset, epochs=5)
        return model
    except Exception as e:
        print("Training failed due to:", str(e))

# ========== Test Case: Model Predicts Next Character ==========
def test_prediction(model, seed, tokenizer, steps=20):
    model.eval()
    input_seq = torch.tensor([tokenizer.encode(seed)])
    h = None
    output = seed
    with torch.no_grad():
        for _ in range(steps):
            logits, h = model(input_seq, h)
            last_logits = logits[:, -1, :]
            next_idx = torch.argmax(last_logits, dim=-1).item()
            next_char = tokenizer.idx2char[next_idx]
            output += next_char
            input_seq = torch.tensor([[next_idx]])
    print(f"Test Generation: {output}")

# ========== Main Script ==========
if __name__ == "__main__":
    print("Training RNN on sample text...")
    model = try_training()
    print("\nTesting model prediction from seeds:")
    if model:
        test_prediction(model, "ROMEO:", tokenizer)
        test_prediction(model, "To be, or", tokenizer)
        test_prediction(model, "JULIET:", tokenizer)
        test_prediction(model, "ACT II. SCENE I.", tokenizer)
        test_prediction(model, "Enter HAMLET", tokenizer)
