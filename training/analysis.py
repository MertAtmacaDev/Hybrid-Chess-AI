import chess
import torch
import pandas as pd
import matplotlib.pyplot as plt
from engine.chess_cnn import ChessCNN, fen_to_tensor
from engine.engine import evaluate, evaluate_cnn



print("=" * 50)
print("analysis 1: PST vs CNN output comparison")
print("=" * 50)

test_positions = {
    "starting": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "white 1 pawn advantage": "rnbqkbnr/ppp1pppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "white queen advantage": "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "black rook advantage": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBN1 w Kkq - 0 1",
    "endgame few pieces": "4k3/8/8/8/8/8/4P3/4K2R w K - 0 1",
}

for name, fen in test_positions.items():
    board = chess.Board(fen)
    pst_score = evaluate(board)
    cnn_score = evaluate_cnn(board)
    print(f"{name}:")
    print(f"  PST: {pst_score:>8.1f}  |  CNN: {cnn_score:>8.1f}")
    print()

print("=" * 50)
print("analysis 2: training data distribution")
print("=" * 50)

# https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations
df = pd.read_csv("chessData.csv", nrows=1000000)
df = df[~df['Evaluation'].str.contains('#')]
df['Evaluation'] = df['Evaluation'].astype(int)
df['Evaluation'] = df['Evaluation'].clip(-5000, 5000)

evals = df['Evaluation']
print(f"mean: {evals.mean():.1f}")
print(f"std dev: {evals.std():.1f}")
print(f"min: {evals.min()}, Max: {evals.max()}")
print(f"between -50 and +50: %{(evals.between(-50, 50).sum() / len(evals) * 100):.1f}")
print(f"between -100 and +100: %{(evals.between(-100, 100).sum() / len(evals) * 100):.1f}")

plt.figure(figsize=(10, 5))
plt.hist(evals, bins=200, edgecolor='black')
plt.title("stockfish eval distribution (centipawn)")
plt.xlabel("eval")
plt.ylabel("frequency")
plt.xlim(-1500, 1500)
plt.savefig("eval_distribution.png")
print("histogram -> eval_distribution.png saved.")