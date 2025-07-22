import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import requests
import nltk
from nltk.tokenize import word_tokenize

# ======== Download NLTK data if needed ========
nltk.download('punkt')

# =============== Sample Dataset (Word-Level) ================
shakespeare_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
sample_text = requests.get(shakespeare_url).text
sample_text = sample_text[:20000]  # Use a subset for fast testing

# ----------------- Tokenizer -----------------
class WordTokenizer:
    def __init__(self, text):
        tokens = word_tokenize(text)
        self.vocab = sorted(set(tokens))
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.tokens = tokens
    def encode(self, words):
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
def generate_text(model, seed, tokenizer, steps=20):
    model.eval()
    input_idxs = tokenizer.encode(seed)
    input_seq = torch.tensor([input_idxs])
    output = word_tokenize(seed)
    with torch.no_grad():
        for _ in range(steps):
            logits = model(input_seq)
            last_logits = logits[0, -1, :]
            next_idx = torch.argmax(last_logits).item()
            next_word = tokenizer.idx2word[next_idx]
            output.append(next_word)
            input_seq = torch.tensor([input_idxs + [next_idx]])
            input_idxs.append(next_idx)
    print(f"Generated: {' '.join(output)}")

# ========== Main ==========
if __name__ == "__main__":
    print("Preparing data...")
    dataset = WordDataset(tokenizer.tokens, tokenizer, seq_len=10)
    vocab_size = len(tokenizer.vocab)
    model = BasicTransformer(vocab_size)
    print("Training transformer...")
    train_model(model, dataset, epochs=5)
    print("\nGenerating text from seed 'ROMEO : '")
    generate_text(model, "ROMEO :", tokenizer, steps=30)
