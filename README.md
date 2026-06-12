# Hybrid-Chess-AI

A chess engine that tests two different ways to evaluate positions: a classic method (Piece-Square Tables) and a deep learning method (CNN). Both use the same search algorithm (Minimax with Alpha-Beta Pruning) — only the evaluation part is different.

## Features

**Search Algorithm:**
- Minimax with Alpha-Beta Pruning
- Quiescence Search (keeps looking at capture moves so it doesn't stop at a bad moment)
- Move Ordering (checks captures first for faster pruning)
- Transposition Table (saves positions that were already checked)

**Evaluation Functions:**
- **PST (Piece-Square Tables):** Each piece has a base value + a bonus based on its position on the board
- **CNN:** A neural network trained on Stockfish scores to guess how good a position is

## Project Structure

```
├── engine/
│   ├── engine.py          # Search algorithm and both evaluation functions
│   └── chess_cnn.py       # CNN model and FEN-to-tensor conversion
├── gui/
│   ├── constants.py       # Piece values, piece-square tables, display settings
│   └── renderer.py        # Board drawing with Pygame
├── scripts/
│   └── compare.py         # Plays games between PST and CNN engines
├── training/
│   ├── train.py           # Training script for the CNN model
│   └── analysis.py        # Checks CNN outputs and looks at data distribution
├── assets/                # Chess piece images
├── models/                # Trained model weights (.pth files)
├── main.py                # Main game loop (play against the engine)
└── requirements.txt
```

## CNN Architecture

- **Input:** 17-channel 8x8 tensor (12 piece planes + turn + 4 castling rights)
- **Layers:** 3x Conv2d + BatchNorm + ReLU (17→32→64→128 channels)
- **Head:** Flatten → FC(8192→256) → ReLU → Dropout(0.3) → FC(256→1)
- **Output:** A single number that shows how good the position is (in centipawns)
- **Training Data:** [Kaggle Chess Evaluation Dataset](https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations)

## Results

All matches are 20 games. The first 6 moves are random.

| Model | PST Wins | Draws | CNN Wins |
|-------|----------|-------|----------|
| 200K samples, 10 epochs, MSE | 19 | 1 | 0 |
| 1M samples, 10 epochs, MSE | 16 | 4 | 0 |
| 1M samples, 20 epochs, BN+Dropout, MSE | 15 | 5 | 0 |
| 1M balanced samples, 20 epochs, Huber Loss | 4 | 15 | 1 |

PST wins every experiment, but the last model (balanced data + Huber Loss) was much better — CNN went from losing almost every game to drawing most of them.

## Key Finding

The CNN was giving ~45 points to almost every position (the dataset average was 39.7). Most of the training data was close to 0, so the model just learned to always guess the average. After we fixed this with balanced sampling (equal number of positions from each score range) and switched from MSE to Huber Loss, the CNN started to tell the difference between good and bad positions.

## How to Run

```bash
# Play against the engine
python main.py

# Train a new CNN model
python -m training.train

# Run PST vs CNN match
python -m scripts.compare

# Check CNN outputs and data distribution
python -m training.analysis
```

## Requirements

```bash
pip install -r requirements.txt
```

## Future Plans

- Speed benchmarks (how fast PST vs CNN makes a move)
- Elo estimation for both engines
- Training with 5M+ samples
