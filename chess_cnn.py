import torch.nn as nn
import numpy

class ChessCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(17, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128*8*8, 256)
        self.fc2 = nn.Linear(256, 1)
        self.bn1 = nn.BatchNorm2d(32)   # conv1'in çıktısı 32
        self.bn2 = nn.BatchNorm2d(64)   # conv2'nin çıktısı 64
        self.bn3 = nn.BatchNorm2d(128)  # conv3'ün çıktısı 128
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
def fen_to_tensor(fen):
    dict_pieces = {
        "P": 0,
        "N": 1,
        "B": 2, 
        "R": 3,
        "Q": 4,
        "K": 5,
        "p": 6,
        "n": 7,
        "b": 8,
        "r": 9,
        "q": 10,
        "k": 11
    }
    
    chess_board = numpy.zeros((17,8,8)) #[kanal][satır][sütun]12+5

    parts = fen.split()
    fen_board = parts[0]
    turn_info = parts[1]
    castling = parts[2]

    if turn_info == "w":
        chess_board[12] = 1

    if "K" in castling:
        chess_board[13] = 1
    if "Q" in castling:
        chess_board[14] = 1
    if "k" in castling:
        chess_board[15] = 1
    if "q" in castling:
        chess_board[16] = 1

    row_fen = fen_board.split("/")
    for i in range(8):
        new_row_fen = row_fen[i]

        col = 0
        for j in range(len(new_row_fen)):
            square_fen = new_row_fen[j]#3, p, 4

            if square_fen.isdigit():
                col += int(square_fen)
            else:
                chess_board[dict_pieces[square_fen], 7-i, col] = 1
                col +=1
    
    return chess_board

