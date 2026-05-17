import pygame
import chess
from gui.constants import SQ_SIZE, BOARD_SIZE

LIGHT = pygame.Color(235, 235, 208)
DARK = pygame.Color(119, 149, 86)

IMAGES = {}

def load_piece_images():
    pieces = ['wp', 'wR', 'wN', 'wB', 'wQ', 'wK', 'bp', 'bR', 'bN', 'bB', 'bQ', 'bK']
    for piece in pieces:
        try:
            IMAGES[piece] = pygame.transform.smoothscale(pygame.image.load(f"assets/{piece}.png"), (SQ_SIZE, SQ_SIZE))
        except FileNotFoundError:
            print(f"image {piece} not found")

def draw_board(screen):
    for row in range(8):
        for col in range(8):
            if (col+row)% 2 == 0:
                color = LIGHT
            else:
                color = DARK
            pygame.draw.rect(screen, color, pygame.Rect(col*SQ_SIZE,row*SQ_SIZE, SQ_SIZE, SQ_SIZE))

def draw_pieces(screen, board: chess.Board):
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color = 'w' if piece.color == chess.WHITE else 'b'

            symbol = piece.symbol().upper()
            piece_type = symbol if symbol != 'P' else 'p'
            image_name = color + piece_type
            
            # I reversed the rows bc the pygame and chess libs are incompatible. Chess -> Pygame
            row = (BOARD_SIZE -1 ) - (square // 8)
            col = square % 8
            
            screen.blit(IMAGES[image_name], pygame.Rect(col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def draw_highlight(screen, selected_square):
    if selected_square != None:
        col = selected_square % 8
        row = 7 - (selected_square // 8)
        
        pygame.draw.rect(screen, (100, 149, 237), (col*SQ_SIZE, row*SQ_SIZE, SQ_SIZE, SQ_SIZE))

def draw_legal_moves(screen, selected_square: int, board: chess.Board):
    if selected_square == None:
        return

    for move in board.legal_moves:
        if move.from_square == selected_square:
            #print(move.to_square)
            row = 7 - (move.to_square // 8)
            col = move.to_square % 8
            center_x = col * SQ_SIZE + SQ_SIZE // 2
            center_y = row * SQ_SIZE + SQ_SIZE // 2
            
            if board.piece_at(move.to_square):
                pygame.draw.circle(screen, (75,75,75), (center_x, center_y), 32, 3)
            else:
                pygame.draw.circle(screen, (75,75,75), (center_x, center_y), 16, 0)