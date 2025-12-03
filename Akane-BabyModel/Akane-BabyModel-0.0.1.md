# 🌸 Akane-BabyModel 0.0.1

**Mini Conversational AI / Chatbot**

---

## Overview

Akane-BabyModel is a lightweight AI chatbot trained on a small dataset of text lines. It generates short, feel-good replies in a conversational style. This version (0.0.1) is designed for **small chats**, giving random-like but coherent responses based on the training data.

It’s powered by a **tiny Transformer-based model** and uses a tokenizer to encode/decode text. Akane aims to provide a **friendly and cute conversation experience**.

---

## Features

- Generates **short, context-based replies**.
- Can run on **CPU or integrated GPU**.
- Mini dataset (~2200 lines) for quick experimentation.
- Separate `chat.py` and `generate.py` for modular design:
  - `chat.py` → interface to talk with Akane.
  - `generate.py` → handles reply generation from the model.
- Lightweight, fast, and fun.

---

## How to use

1. Train the model (if not already trained):
   ```bash
   python train_model.py
2. Chat with Akane:

python chat.py
3. Example Conversation 

🌸 Akane is ready! Type 'exit' to quit.
You: hi
Akane: topic playfully better
You: wow
Akane: without pastels
You: super
Akane: Awesome your
You: exit
🌸 Akane: Bye! We Can Talk soon 💖

## Notes

This is version 0.0.1, a prototype.

Akane generates short sentences that may not always fully answer questions.

Future versions can include more training data, longer conversations, and better coherence.