import chess

PieceWorth = {
    chess.PAWN: 100,
    chess.KNIGHT:300,
    chess.BISHOP:325,
    chess.ROOK:500,
    chess.QUEEN:900,
    chess.KING:30000
}

def evaluate(board: chess.Board):
    total_score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece == None:
            continue
        p_type=piece.piece_type
        p_color = piece.color
        
        if p_color == chess.WHITE:
            total_score += PieceWorth[p_type]
        else:
            total_score -= PieceWorth[p_type]

    return total_score

def minimax(board: chess.Board, depth, maximing, alpha, beta):
    
    if depth == 0:
        return evaluate(board)
    
    legal_moves =list(board.legal_moves)
    if not legal_moves:
        return evaluate(board)
    

    if maximing  == True:#White
        best_score = float('-inf')
        for move in legal_moves:
            board.push(move)
            current_score = minimax(board, depth-1, False, alpha, beta)
            board.pop()

            best_score = max(best_score, current_score)
            alpha = max(alpha, best_score)

            if alpha >=beta:
                break

        return best_score
    
    if maximing == False:#Black
        best_score = float('inf')
        for move in legal_moves:
            board.push(move)
            current_score = minimax(board, depth-1, True, alpha, beta)
            board.pop()
            
            best_score = min(best_score, current_score)
            beta = min(beta, best_score)

            if beta <= alpha:
                break
            
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

    for move in board.legal_moves:
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
