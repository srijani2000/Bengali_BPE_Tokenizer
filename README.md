# 🇧🇩 Bengali BPE Tokenizer

A high-performance **Byte Pair Encoding (BPE)** tokenizer specifically designed for Bengali text, achieving exceptional compression ratios while maintaining linguistic accuracy.


---

## ✨ Key Features

- **🎯 High Compression Ratio**: Achieves **5.09x** compression (characters to tokens)
- **📚 Large Vocabulary**: **6,000** token vocabulary size
- **🔄 Efficient Merges**: **5,568** BPE merge operations
- **🌐 Bengali-Optimized**: Built specifically for Bengali Unicode characters (U+0980 - U+09FF)
- **🚀 Production Ready**: Includes Gradio web interface for interactive tokenization
- **💾 Easy Integration**: Simple save/load functionality for model persistence

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Compression Ratio** | **5.09** |
| **Vocabulary Size** | **6,000** tokens |
| **Total Merges** | **5,568** operations |
| **Training Corpus** | Kaggle Bengali Dataset |
| **Corpus Size** | 2.76M characters, 19,582 lines |

---

## 🏗️ Architecture

The tokenizer implements a custom BPE algorithm optimized for Bengali text:

1. **Base Vocabulary**: Initializes with all Bengali Unicode characters (128 characters) + whitespace
2. **Iterative Merging**: Progressively merges most frequent character pairs
3. **Frequency Threshold**: Uses `min_freq=2` to ensure statistical significance
4. **Special Tokens**: Includes `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>` for downstream tasks
5. **REGEX Used** : (r""" ?[\u0980-\u09FF]+| ?[^\s]+|\s+(?!\S)|\s+""")


The repository includes a **Gradio** web app for interactive tokenization:
python app.py
```
Access the live demo on [Hugging Face Spaces](https://huggingface.co/spaces/srijani2/bengali_tokenizer) 🚀

---

## 📁 Project Structure

```
.
├── bengali_bpe_tokenizer.py    # Core tokenizer implementation
├── bengali_bpe_tokenizer.json  # Pre-trained model (6,000 vocab)
├── app.py                      # Gradio web interface
├── article.txt                 # Training corpus (Kaggle dataset)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 📈 Training Details

### Dataset
- **Source**: Kaggle Bengali Text Dataset
- **Size**: 2,764,226 characters
- **Lines**: 19,582
- **Words**: 401,281
- **Bengali Content**: 84.25% Bengali characters

### Training Configuration
- **Vocabulary Size**: 6,000 tokens
- **Minimum Frequency**: 2 (pairs must appear at least twice)
- **Base Vocabulary**: 128 Bengali characters + 3 whitespace tokens
- **Special Tokens**: 4 (`<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`)

### Training Process
1. Text is tokenized using Bengali-specific regex patterns
2. Words are split into character sequences
3. Most frequent character pairs are iteratively merged
4. Process continues until vocabulary size reaches 6,000 or no more valid pairs exist

---

