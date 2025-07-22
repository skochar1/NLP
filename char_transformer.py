import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import requests

# =============== Sample Dataset (Character-Level) ================
shakespeare_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
sample_text = requests.get(shakespeare_url).text
sample_text = sample_text[:10000] # Use a subset for fast testing

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
    def __init__(self, text, tokenizer, seq_len=40):
        self.data = tokenizer.encode(text)
        self.seq_len = seq_len
    def __len__(self):
        return len(self.data) - self.seq_len
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+1:idx+self.seq_len+1]
        return torch.tensor(x), torch.tensor(y)

# ============ Transformer Model Definition ==============
class BasicTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, nhead=4, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, 1000, embed_dim)) # max sequence length 1000
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)
    def forward(self, x):
        x = self.embed(x) + self.pos_enc[:, :x.size(1), :]
        x = x.transpose(0, 1)  # transformer expects [seq, batch, embed]
        out = self.transformer(x)
        out = out.transpose(0, 1)
        logits = self.fc(out)
        return logits

# ========== Training Loop ==========
def train_model(model, dataset, epochs=5, lr=0.001):
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in dataloader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")

# ========== Generation ==========
def generate_text(model, seed, tokenizer, steps=50):
    model.eval()
    input_idxs = tokenizer.encode(seed)
    input_seq = torch.tensor([input_idxs])
    output = seed
    with torch.no_grad():
        for _ in range(steps):
            logits = model(input_seq)
            last_logits = logits[0, -1, :]
            next_idx = torch.argmax(last_logits).item()
            next_char = tokenizer.idx2char[next_idx]
            output += next_char
            input_seq = torch.tensor([input_idxs + [next_idx]])
            input_idxs.append(next_idx)
    print(f"Generated: {output}")

# ========== Main ==========
if __name__ == "__main__":
    print("Preparing data...")
    dataset = CharDataset(sample_text, tokenizer, seq_len=40)
    vocab_size = len(tokenizer.chars)
    model = BasicTransformer(vocab_size)
    print("Training transformer...")
    train_model(model, dataset, epochs=5)
    print("\nGenerating text from seed 'ROMEO: '")
    generate_text(model, "ROMEO: ", tokenizer, steps=100)
