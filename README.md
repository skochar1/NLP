 Text Generation with RNN/LSTM/Transformer: Character-Level vs Word-Level

This project demonstrates **text generation** using PyTorch, showcasing both **character-level** and **word-level** sequence models, implemented with both traditional RNN/LSTM and Transformer architectures.

It includes:
- A character-level RNN/LSTM (`char_rnn.py`)
- A word-level RNN/LSTM (`word_rnn.py`)
- A character-level Transformer (`char_transformer.py`)
- A word-level Transformer (`word_transformer.py`)

The [Tiny Shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) is used as the default data source for all models.

---

## Features

- **Tokenizer:** Character-level or word-level tokenization
- **Custom PyTorch Dataset** for efficient training
- **Architectures:**
  - Basic RNN/LSTM
  - 2-layer bidirectional LSTM with dropout (default)
  - Basic Transformer Encoder
- **Flexible training loop**
- **Text generation** from a custom seed
- **Easy customization** for data, model size, training time, etc.

---

## File Structure

```
char_rnn.py           # Character-level RNN/LSTM text generator
word_rnn.py           # Word-level RNN/LSTM text generator
char_transformer.py   # Character-level Transformer text generator
word_transformer.py   # Word-level Transformer text generator
```

---

## Getting Started

### 1. **Install Requirements**

Make sure you have Python 3.8+ and [PyTorch](https://pytorch.org/get-started/locally/).
You’ll also need nltk and requests:

```bash
pip install torch requests nltk
```

### 2. **Run the Scripts**

#### For character-level RNN/LSTM:
```bash
python char_rnn.py
```

#### For word-level RNN/LSTM:
```bash
python word_rnn.py
```

#### For character-level Transformer:
```bash
python char_transformer.py
```

#### For word-level Transformer:
```bash
python word_transformer.py
```

---

## How the Code Works

### **1. Tokenizer**
- **char_rnn.py / char_transformer.py:**  
  Tokenizes at the **character** level. Each character is mapped to an integer index.
- **word_rnn.py / word_transformer.py:**  
  Tokenizes at the **word** level using NLTK's word_tokenize. Each word is mapped to an integer index.

### **2. Dataset**
- All scripts create sequences of tokens (seq_len long) and train the model to predict the next token (character or word).

### **3. Model**
- **RNN/LSTM:**  
  RNN or LSTMRNN with embedding, LSTM (or basic RNN), and a fully connected output layer.
  LSTMRNN is a 2-layer bidirectional LSTM by default.
- **Transformer:**  
  Embedding + positional encoding, multi-layer Transformer encoder, and a fully connected output layer.  
  Uses self-attention instead of hidden state memory.

### **4. Training**
- All scripts use cross-entropy loss and Adam optimizer.
- Training prints the average loss per epoch.

### **5. Text Generation**
- Given a starting "seed" (word or character(s)), the model predicts and appends the next token, generating new text.
- In word-level, output grows word by word; in char-level, it grows character by character.

---

## Example Output

**char_rnn.py** (character-level, RNN/LSTM)
```
Training RNN on sample text...
Epoch 1 Loss: 2.3104
...
Testing model prediction from seed 'hello':
Test Generation: hellos that so the sain...
```

**word_rnn.py** (word-level, RNN/LSTM)
```
Training LSTM on sample text...
Epoch 1 Loss: 2.8497
...
Testing model prediction from seed "ROMEO :"
Test Generation: ROMEO : Thou : yet : yet : suffer suffer ...
```

**char_transformer.py** (character-level, Transformer)
```
Training transformer...
Epoch 1 Loss: 2.2
...
Generating text from seed 'ROMEO: '
Generated: ROMEO: the king the king the king the king the king...
```

**word_transformer.py** (word-level, Transformer)
```
Training transformer...
Epoch 1 Loss: 2.7
...
Generating text from seed 'ROMEO : '
Generated: ROMEO : king king king king king ...
```

---

## RNN/LSTM vs Transformer: Key Differences

| Aspect              | RNN/LSTM                                     | Transformer                                      |
|---------------------|----------------------------------------------|--------------------------------------------------|
| **Sequence Handling** | Processes one token at a time (sequentially) | Processes the whole sequence in parallel using self-attention |
| **Context Modeling**  | Remembers context via hidden state            | Uses self-attention to relate all positions to each other      |
| **Parallelism**       | Low (one step after another)                  | High (parallel over sequence positions)          |
| **Long-Range Dependencies** | Weaker (can forget distant info)       | Strong (directly models relationships between all tokens)      |
| **Positional Information** | Implicit (via order of input)           | Explicit (via positional encodings)              |
| **Training Speed**    | Slower for long sequences                    | Faster on modern hardware                        |

---

## Character-level vs Word-level: Key Differences

| Aspect          | char_rnn.py / char_transformer.py         | word_rnn.py / word_transformer.py                |
|-----------------|-------------------------------------------|--------------------------------------------------|
| **Unit**        | Character (a, b, c, ., :, etc.)           | Word (“ROMEO”, “:”, “Thou”, etc.)                |
| **Tokenizer**   | Each character gets an index               | Each word gets an index (uses nltk.word_tokenize) |
| **Vocab Size**  | Small (e.g. 60-100 unique chars)           | Large (1000+ unique words even in small dataset) |
| **Input/Output**| Sequence of chars → next char              | Sequence of words → next word                    |
| **Granularity** | Learns spelling, punctuation, etc.         | Learns word order, word collocations, etc.       |
| **Generated Text** | May create new words, creative outputs  | Always real words, but may repeat short phrases  |
| **Data Needs**  | Works with less data, but slow to learn structure | Needs much more data to generalize           |

**In summary:**  
- Use **char_rnn.py** or **char_transformer.py** for fine-grained, creative, or noisy text modeling (or if you have very little data).
- Use **word_rnn.py** or **word_transformer.py** for more natural word sequences and if you have more data/memory.
- **Transformers** are now the industry standard for all large-scale NLP.

---

## Tips for Better Results

- **Train longer and on more data** for all models.
- **Increase model size** (embedding and hidden units) for better results if your hardware allows.
- **Try different seeds and temperature values** during generation.
- **For truly realistic English output, use subword tokenization (e.g., with HuggingFace) and larger transformer models.**

---

## References

- [Tiny Shakespeare dataset by Andrej Karpathy](https://github.com/karpathy/char-rnn)
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html)
- [NLTK documentation](https://www.nltk.org/)

---

## License

This project is for educational purposes and leverages publicly available datasets and open-source libraries.  
Original Tiny Shakespeare dataset by Andrej Karpathy ([MIT License](https://github.com/karpathy/char-rnn/blob/master/LICENSE)).
