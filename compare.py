import chess
import engine

import random

def play_game(white_uses_cnn, black_uses_cnn, depth=3, random_opening_moves=6):
    board = chess.Board()
    
    for _ in range(random_opening_moves):
        legal = list(board.legal_moves)
        if not legal:
            break
        board.push(random.choice(legal))
    
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            engine.USE_CNN = white_uses_cnn
        else:
            engine.USE_CNN = black_uses_cnn

        move = engine.minimax_move(board, depth)
        board.push(move)

    return board.result()

for i in range(5):
    result = play_game(False, True)
    print(f"Oyun {i+1}: {result}")

for i in range(5):
    result = play_game(True, False)
    print(f"Oyun {i+1}: {result}")