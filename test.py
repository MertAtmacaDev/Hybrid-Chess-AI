"""import chess
import engine
import numpy

test=engine.fen_to_tensor("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
print("Beyaz piyon (kanal 0), satır 1:", test[0][1])
print("Siyah piyon (kanal 6), satır 6:", test[6][6])
print("Beyaz kale (kanal 3):", test[3][0][0], test[3][0][7])
print("Sıra kanalı (12) toplamı:", numpy.sum(test[12]))
print("Toplam:", numpy.sum(test))"""

import torch
print(torch.__version__)