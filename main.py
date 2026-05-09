import pygame
import chess
import sys
import renderer
import engine
import constants

def main():
    pygame.init()
    screen = pygame.display.set_mode((constants.WIDTH, constants.HEIGHT))
    pygame.display.set_caption("Hybrid-Chess-AI")
    clock = pygame.time.Clock()
    
    board = chess.Board()
    renderer.load_piece_images()
    
    running = True
    selected_square = None
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    #score = engine.minimax(board, depth=2, maximing=True)
                    #print(f"Skor: {score}")

                    move = engine.minimax_move(board,depth=3)
                    print(move)
                
            if event.type == pygame.MOUSEBUTTONDOWN:
                x,y = pygame.mouse.get_pos()
                col = x // constants.SQ_SIZE
                row = y // constants.SQ_SIZE

                
                square = (7-row)*8 + col
                piece = board.piece_at(square)

                if selected_square == None:
                    if piece != None:
                        selected_square = square
                        print(f"Selected {square}")
                else:
                    new_move = chess.Move(selected_square, square)

                    if new_move in board.legal_moves:
                        board.push(new_move)
                        print("new move correct")
                    else:
                        print("illegal move")


                    selected_square = None

                print(f"Click for pygame. px(x = {x}, y = {y}), row({row}), col({col}) square({piece})")# (7-row)*8+col formula check

        renderer.draw_board(screen)
        renderer.draw_highlight(screen, selected_square)
        renderer.draw_pieces(screen, board)
        renderer.draw_legal_moves(screen, selected_square, board)
        
        
        pygame.display.flip()
        clock.tick(constants.FPS)
        
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()