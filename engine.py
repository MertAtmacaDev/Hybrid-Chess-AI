import chess
from collections import deque
from constants import PieceWorth, PST
import numpy
from chess_cnn import ChessCNN
import torch

cnn_model = ChessCNN()
cnn_model.load_state_dict(torch.load("model.pth"))
cnn_model.eval()

USE_CNN = True

transposition_table = {} #transposition_table values = [best_score][depth][flag]

exact = 3
lowerbound = 2
upperbound = 1

def evaluate(board: chess.Board):
    total_score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece == None:
            continue
        p_type=piece.piece_type
        p_color = piece.color
        
        if p_color == chess.WHITE:
            row = square // 8
            col = square % 8
            new_square = (7-row)*8+col

            bonus = PST[p_type][new_square]
            total_score += PieceWorth[p_type] + bonus
        else:
            bonus = PST[p_type][square]
            total_score -= PieceWorth[p_type] + bonus

    return total_score

def evaluate_cnn(board):
    tensor = fen_to_tensor(board.fen())
    tensor = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        score = cnn_model(tensor)
    return score.item()*100

def move_ordering(board: chess.Board):

    legal_moves=list(board.legal_moves)
    new_legal_moves = deque()

    for move in legal_moves:
        if board.is_capture(move):
            new_legal_moves.appendleft(move)
        else:
            new_legal_moves.append(move)
    
    return new_legal_moves

def quiescence(board: chess.Board, alpha, beta, maximizing):
    if USE_CNN:
        best_score = evaluate_cnn(board)
    else:
        best_score = evaluate(board)
    
    if maximizing == True:
        alpha = max(alpha, best_score)
        if alpha >= beta:
            return best_score
        
        for move in board.legal_moves:
            if board.is_capture(move):
                board.push(move)
                current_score = quiescence(board, alpha, beta, False)
                board.pop()

                best_score = max(best_score, current_score)
                alpha = max(alpha, best_score)

                if alpha >= beta:
                    return best_score
        return best_score
    else:
        beta = min(beta, best_score)
        if beta <= alpha:
            return best_score
        
        for move in board.legal_moves:
            if board.is_capture(move):
                board.push(move)
                current_score = quiescence(board, alpha, beta, True)
                board.pop()
                
                best_score = min(best_score, current_score)
                beta = min(beta, best_score)

                if beta <= alpha:
                    return best_score
    
        return best_score

def minimax(board: chess.Board, depth, maximizing, alpha, beta):
    if depth == 0:
        return quiescence(board, alpha, beta, (board.turn == chess.WHITE))
    
    legal_moves =move_ordering(board)
    if not legal_moves:
        if board.is_checkmate():
            if maximizing:
                return -99999 - depth
            else:
                return 99999 + depth
        else:
            return 0
    
    current_board = board.fen()
    if current_board in transposition_table:
        if transposition_table[current_board][1] >= depth:
            if transposition_table[current_board][2] == exact:
                return transposition_table[current_board][0]
            if transposition_table[current_board][2] == lowerbound:
                if transposition_table[current_board][0] >= beta:
                    return transposition_table[current_board][0]
            if transposition_table[current_board][2] == upperbound:
                if transposition_table[current_board][0] <= alpha:
                    return transposition_table[current_board][0]
            
    if maximizing  == True:#White
        best_score = float('-inf')
        flag = exact
        original_alpha = alpha
        for move in legal_moves:
            board.push(move)

            current_score = minimax(board, depth-1, False, alpha, beta)
            board.pop()
            
            best_score = max(best_score, current_score)
            alpha = max(alpha, best_score)

            if alpha >=beta:
                flag = lowerbound
                break
        if best_score <= original_alpha:
            flag = upperbound

        if (current_board not in transposition_table or
             transposition_table[current_board][1] < depth or
             (transposition_table[current_board][1] == depth and flag > transposition_table[current_board][2])):
            transposition_table[current_board] = (best_score, depth, flag)
        return best_score
    
    if maximizing == False:#Black
        best_score = float('inf')
        flag = exact
        original_beta = beta
        for move in legal_moves:
            board.push(move)
            current_score = minimax(board, depth-1, True, alpha, beta)
            board.pop()
            
            best_score = min(best_score, current_score)
            beta = min(beta, best_score)

            if beta <= alpha:
                flag = upperbound
                break
        
        if best_score >= original_beta:
            flag = lowerbound

        if (current_board not in transposition_table or
             transposition_table[current_board][1] < depth or
             (transposition_table[current_board][1] == depth and flag > transposition_table[current_board][2])):
            transposition_table[current_board] = (best_score, depth, flag)
        return best_score

def minimax_move(board: chess.Board, depth):
    best_move = None
    alpha = float('-inf')
    beta = float('inf')
    is_white_turn = (board.turn == chess.WHITE)

    if is_white_turn:
        best_score = float('-inf')
    else:
        best_score = float('inf')

    for move in move_ordering(board):
        board.push(move)
        current_score = minimax(board, depth-1, not(is_white_turn), alpha, beta)

        if is_white_turn:
            if current_score > best_score:
                best_score = current_score
                best_move = move
            alpha = max(alpha, best_score)
        else:
            if current_score < best_score:
                best_score = current_score
                best_move = move
            beta = min(beta, best_score)
        board.pop()
    
    return best_move

def fen_to_tensor(fen):#12x8x8 boyutunda 0 ve 1'lerden oluşan bir yapı. 
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

