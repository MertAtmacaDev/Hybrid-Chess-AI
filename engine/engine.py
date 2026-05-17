import chess
from collections import deque
from gui.constants import PieceWorth, PST
import numpy
from engine.chess_cnn import ChessCNN
import torch
from engine.chess_cnn import ChessCNN, fen_to_tensor

cnn_model = ChessCNN()
cnn_model.load_state_dict(torch.load("models/model-balanced-huber-5m.pth"))
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
