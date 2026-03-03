import pygame
import chess
from bot_VIP import NNBt

# ─────────────────────────────
# CONFIG
# ─────────────────────────────
MODEL_PATH = "chess_model.pth"
BOT_DEPTH = 3

WIDTH = HEIGHT = 640
SQ_SIZE = WIDTH // 8

WHITE = (240, 217, 181)
BROWN = (181, 136, 99)
HIGHLIGHT = (100, 200, 100)

# ─────────────────────────────
# LOAD BOT
# ─────────────────────────────
bot = NNBt(MODEL_PATH)
board = chess.Board()

# ─────────────────────────────
# PYGAME INIT
# ─────────────────────────────
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Play vs Neural Chess Bot")
font = pygame.font.SysFont("segoeuisymbol", 52)

selected_square = None

# Unicode pieces
UNICODE_PIECES = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}


# ─────────────────────────────
# DRAW BOARD
# ─────────────────────────────
def draw_board():
    for row in range(8):
        for col in range(8):
            color = WHITE if (row + col) % 2 == 0 else BROWN
            pygame.draw.rect(screen, color,
                             (col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))


def draw_pieces():
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - (square // 8)
            col = square % 8
            text = font.render(
                UNICODE_PIECES[piece.symbol()],
                True,
                (0, 0, 0)
            )
            text_rect = text.get_rect(center=(
                col * SQ_SIZE + SQ_SIZE // 2,
                row * SQ_SIZE + SQ_SIZE // 2
            ))
            screen.blit(text, text_rect)


def highlight_square(square):
    if square is None:
        return
    row = 7 - (square // 8)
    col = square % 8
    pygame.draw.rect(screen, HIGHLIGHT,
                     (col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE), 5)


# ─────────────────────────────
# BOT MOVE
# ─────────────────────────────
def bot_move():
    if not board.is_game_over():
        move = bot.get_best_move(board, depth=BOT_DEPTH)
        if move:
            board.push(move)


# ─────────────────────────────
# MAIN LOOP
# ─────────────────────────────
running = True
player_color = chess.WHITE  # bạn chơi trắng

while running:

    draw_board()
    highlight_square(selected_square)
    draw_pieces()
    pygame.display.flip()

    # Bot đi nếu tới lượt
    if board.turn != player_color and not board.is_game_over():
        bot_move()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN and board.turn == player_color:
            x, y = pygame.mouse.get_pos()
            col = x // SQ_SIZE
            row = 7 - (y // SQ_SIZE)
            square = chess.square(col, row)

            if selected_square is None:
                if board.piece_at(square) and board.piece_at(square).color == player_color:
                    selected_square = square
            else:
                move = chess.Move(selected_square, square)

                # auto queen promotion
                if move in board.legal_moves:
                    board.push(move)
                else:
                    move = chess.Move(selected_square, square, promotion=chess.QUEEN)
                    if move in board.legal_moves:
                        board.push(move)

                selected_square = None

pygame.quit()