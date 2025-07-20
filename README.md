# Character-Level RNN/LSTM Text Generation

This project implements a **character-level text generator** using PyTorch. It includes a basic RNN and a more advanced bidirectional LSTM to learn and generate text, with the [Tiny Shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) as the default training data.

## Features

- **Character-level tokenizer** (learns at the character, not word, level)
- **Custom PyTorch Dataset** for training samples
- Two architectures:
  - Basic RNN
  - 2-layer bidirectional LSTM with dropout (default)
- **Training loop with options for fine-tuning**
- **Test generation** from a text seed
- **Simple API** for swapping architectures or using your own data

---

## Getting Started

### 1. **Install Requirements**

Make sure you have Python 3.8+ and [PyTorch](https://pytorch.org/get-started/locally/).  
You can install dependencies with:

```bash
pip install torch requests
```

### 2. **Run the Script**

Save the script as `rnn_class.py` and run:

```bash
python rnn_shakespeare.py
```

It will:
- Download the Tiny Shakespeare dataset
- Train the LSTM model on the data for 5 epochs (default)
- Generate text from the seed `"hello"`

---

## File Structure

```
rnn_class.py        # Main training and generation script
```

---

## How It Works

- **Tokenizer:** Maps every unique character in the training text to an integer and vice versa.
- **Dataset:** Creates training pairs of (input_sequence, target_sequence), where each sequence is a list of character indices.
- **Models:**  
  - `RNN`: Simple single-layer RNN.
  - `LSTMRNN`: More powerful 2-layer bidirectional LSTM with dropout.
- **Training:** Learns to predict the next character for each position in a sequence.
- **Generation:** Given a seed, predicts the next character repeatedly to form new text.

---

## Customization

- **Change the model:**  
  Switch between `RNN` and `LSTMRNN` in `try_training()`.
- **Use your own data:**  
  Replace the `sample_text` loading step with any large string.
- **Adjust sequence length:**  
  Change the `seq_len` parameter in `CharDataset`.
- **Change training parameters:**  
  Update `epochs`, `lr`, or model sizes in the respective functions/classes.

---

## Example Output

```
Training RNN on sample text...
Epoch 1 Loss: 2.3104
Epoch 2 Loss: 1.5315
...
Testing model prediction from seed 'hello':
Test Generation: hellos that so the sain...
```

---

## Tips for Better Results

- **Use longer training (more epochs) for better text.**
- **Try more creative generation** by sampling from the output distribution (add a temperature parameter and sample instead of using `argmax`).
- **Train on more or different text** for more variety and creativity.
- **Upgrade your architecture** (try more layers, larger hidden sizes, etc.).

---

## References

- [Tiny Shakespeare dataset by Andrej Karpathy](https://github.com/karpathy/char-rnn)
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html)

---

## License

This project is for educational purposes and leverages publicly available datasets and open-source libraries.  
Original Tiny Shakespeare dataset by Andrej Karpathy ([MIT License](https://github.com/karpathy/char-rnn/blob/master/LICENSE)).
