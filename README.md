# Text Generation with RNN/LSTM: Character-Level vs Word-Level

This project demonstrates **text generation** using PyTorch, showcasing both **character-level** and **word-level** sequence models. It includes:
- A character-level RNN/LSTM (`char_rnn.py`)
- A word-level RNN/LSTM (`word_rnn.py`)

The [Tiny Shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) is used as the default data source for both.

---

## Features

- **Tokenizer:** Character-level or word-level tokenization
- **Custom PyTorch Dataset** for efficient training
- **Two architectures:**
  - Basic RNN
  - 2-layer bidirectional LSTM with dropout (default)
- **Flexible training loop**
- **Text generation** from a custom seed
- **Easy customization** for data, model size, training time, etc.

---

## File Structure

```
char_rnn.py         # Character-level text generator
word_rnn.py         # Word-level text generator
```

---

## Getting Started

### 1. **Install Requirements**

Make sure you have Python 3.8+ and [PyTorch](https://pytorch.org/get-started/locally/).  
You’ll also need `nltk` and `requests`:

```bash
pip install torch requests nltk
```

### 2. **Run the Script**

#### For character-level generation:
```bash
python char_rnn.py
```

#### For word-level generation:
```bash
python word_rnn.py
```

---

## How the Code Works

### **1. Tokenizer**
- **char_rnn.py:**  
  Tokenizes at the **character** level. Each character is mapped to an integer index.
- **word_rnn.py:**  
  Tokenizes at the **word** level using NLTK's `word_tokenize`. Each word is mapped to an integer index.

### **2. Dataset**
- Both scripts create sequences of tokens (`seq_len` long) and train the model to predict the next token (character or word).

### **3. Model**
- `RNN` or `LSTMRNN` is used in both, with embedding, LSTM (or basic RNN), and a fully connected output layer.
- `LSTMRNN` is a 2-layer bidirectional LSTM by default.

### **4. Training**
- Both scripts use cross-entropy loss and Adam optimizer.
- Training prints the average loss per epoch.

### **5. Text Generation**
- Given a starting "seed" (word or character(s)), the model predicts and appends the next token, generating new text.
- In word-level, output grows word by word; in char-level, it grows character by character.

---

## Customization

- **Change the model:**  
  Swap `RNN` and `LSTMRNN` in the main script.
- **Use your own data:**  
  Change the `sample_text` assignment.
- **Adjust sequence length:**  
  Change the `seq_len` parameter.
- **Modify training settings:**  
  Change `epochs`, `lr`, batch size, etc.

---

## Example Output

**char_rnn.py** (character-level)
```
Training RNN on sample text...
Epoch 1 Loss: 2.3104
...
Testing model prediction from seed 'hello':
Test Generation: hellos that so the sain...
```

**word_rnn.py** (word-level)
```
Training LSTM on sample text...
Epoch 1 Loss: 2.8497
...
Testing model prediction from seed "ROMEO :"
Test Generation: ROMEO : Thou : yet : yet : suffer suffer ...
```

---

## Character RNN vs Word RNN: Key Differences

| Aspect          | char_rnn.py                                 | word_rnn.py                                      |
|-----------------|---------------------------------------------|--------------------------------------------------|
| **Unit**        | Character (a, b, c, ., :, etc.)             | Word (“ROMEO”, “:”, “Thou”, etc.)                |
| **Tokenizer**   | Each character gets an index                 | Each word gets an index (uses `nltk.word_tokenize`) |
| **Vocab Size**  | Small (e.g. 60-100 unique chars)             | Large (1000+ unique words even in small dataset) |
| **Input/Output**| Sequence of chars → next char                | Sequence of words → next word                    |
| **Granularity** | Learns spelling, punctuation, etc.           | Learns word order, word collocations, etc.       |
| **Generated Text** | May create new words, creative outputs    | Always real words, but may repeat short phrases  |
| **Data Needs**  | Works with less data, but slow to learn structure | Needs much more data to generalize             |

**In summary:**  
- Use **char_rnn.py** for fine-grained, creative, or noisy text modeling (or if you have very little data).
- Use **word_rnn.py** for more natural word sequences and if you have more data/memory.

---

## Tips for Better Results

- **Train longer and on more data** for both models.
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
