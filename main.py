import pygame
import chess
import sys

WIDTH = 512
HEIGHT = 512
SQ_SIZE = WIDTH // 8
MAX_FPS = 60
BOARD_SIZE = 8
COLORS = [pygame.Color(235, 235, 208), pygame.Color(119, 149, 86)] 

IMAGES = {}

def load_images():
    pieces = ['wp', 'wR', 'wN', 'wB', 'wQ', 'wK', 'bp', 'bR', 'bN', 'bB', 'bQ', 'bK']
    for piece in pieces:
        try:
            IMAGES[piece] = pygame.transform.smoothscale(pygame.image.load(f"images/{piece}.png"), (SQ_SIZE, SQ_SIZE))
        except FileNotFoundError:
            print(f"Uyarı: images/{piece}.png bulunamadı")

def draw_board(screen):
    for row in range(8):
        for col in range(8):
            color = COLORS[(row + col) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def draw_pieces(screen, board):
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color = 'w' if piece.color == chess.WHITE else 'b'

            symbol = piece.symbol().upper()
            piece_type = symbol if symbol != 'P' else 'p'
            image_name = color + piece_type
            
            # python-chess has a1 at the bottom-left, but pygame starts at the top-left. Inverting the row
            row = (BOARD_SIZE -1 ) - (square // 8)
            col = square % 8
            
            screen.blit(IMAGES[image_name], pygame.Rect(col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Hibrit Satranç Yapay Zekası")
    clock = pygame.time.Clock()
    
    board = chess.Board()
    load_images()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        draw_board(screen)
        draw_pieces(screen, board)
        
        pygame.display.flip()
        clock.tick(MAX_FPS)
        
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()