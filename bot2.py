import chess
import chess.polyglot
import numpy as np

INF = 999999

# Giá trị quân cờ
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# ================== EVAL CONSTANTS ==================

BISHOP_PAIR_BONUS = 40

DOUBLED_PAWN_PENALTY = 15
ISOLATED_PAWN_PENALTY = 20
PASSED_PAWN_BONUS = 25

MOBILITY_WEIGHT = 5
KING_SHIELD_BONUS = 15

# PST tối giản cho Mã và Tốt để bot có định hướng
PST = {
    chess.PAWN: [
        0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5,  5, 10, 25, 25, 10,  5,  5,
        0,  0,  0, 20, 20,  0,  0,  0,
        5, -5,-10,  0,  0,-10, -5,  5,
        5, 10, 10,-20,-20, 10, 10,  5,
        0,  0,  0,  0,  0,  0,  0,  0
    ],
    chess.KNIGHT: [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
    ]
}

class NNBt:
    def __init__(self, model_path=None):
        self.tt = {}
        self.history = np.zeros((64, 64), dtype=int)
        self.killers = [[None, None] for _ in range(64)]

    def evaluate(self, board):
        if board.is_checkmate():
            return -INF

        score = 0

        # ================= MATERIAL =================
        for pt, val in PIECE_VALUES.items():
            score += len(board.pieces(pt, chess.WHITE)) * val
            score -= len(board.pieces(pt, chess.BLACK)) * val

        # ================= KNIGHT PST =================
        for sq in board.pieces(chess.KNIGHT, chess.WHITE):
            score += PST[chess.KNIGHT][sq]
        for sq in board.pieces(chess.KNIGHT, chess.BLACK):
            score -= PST[chess.KNIGHT][sq ^ 56]

        # ================= BISHOP PAIR =================
        if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
            score += 35
        if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
            score -= 35

        # ================= PAWN STRUCTURE (FAST) =================
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)

        white_files = [0]*8
        black_files = [0]*8

        for sq in white_pawns:
            white_files[chess.square_file(sq)] += 1
        for sq in black_pawns:
            black_files[chess.square_file(sq)] += 1

        # ---- WHITE ----
        for sq in white_pawns:
            file = chess.square_file(sq)
            penalty = 0

            # doubled
            if white_files[file] > 1:
                penalty += 12

            # isolated
            if (file == 0 or white_files[file-1] == 0) and \
            (file == 7 or white_files[file+1] == 0):
                penalty += 15

            score -= penalty

        # ---- BLACK ----
        for sq in black_pawns:
            file = chess.square_file(sq)
            penalty = 0

            if black_files[file] > 1:
                penalty += 12

            if (file == 0 or black_files[file-1] == 0) and \
            (file == 7 or black_files[file+1] == 0):
                penalty += 15

            score += penalty

        # ================= KING SAFETY (ULTRA LIGHT) =================
        for color in [chess.WHITE, chess.BLACK]:
            king_sq = board.king(color)
            if king_sq is None:
                continue

            file = chess.square_file(king_sq)
            rank = chess.square_rank(king_sq)
            direction = 1 if color == chess.WHITE else -1

            shield = 0
            for df in (-1, 0, 1):
                f = file + df
                r = rank + direction
                if 0 <= f <= 7 and 0 <= r <= 7:
                    sq = chess.square(f, r)
                    piece = board.piece_at(sq)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        shield += 8

            score += shield if color == chess.WHITE else -shield

        return score if board.turn == chess.WHITE else -score

    def order_moves(self, board, depth, hash_move=None):
        moves = list(board.legal_moves)
        d_idx = min(depth, 63)
        
        def score_move(move):
            if move == hash_move: return 1000000
            if board.is_capture(move):
                vic = board.piece_at(move.to_square)
                atker = board.piece_at(move.from_square)
                if vic and atker:
                    return 900000 + (PIECE_VALUES[vic.piece_type] * 10 - PIECE_VALUES[atker.piece_type])
            if move in self.killers[d_idx]: return 800000
            return self.history[move.from_square][move.to_square]

        moves.sort(key=score_move, reverse=True)
        return moves

    def quiescence(self, board, alpha, beta):
        stand_pat = self.evaluate(board)
        if stand_pat >= beta: return beta
        if alpha < stand_pat: alpha = stand_pat

        for move in board.generate_legal_captures():
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha)
            board.pop()
            if score >= beta: return beta
            if score > alpha: alpha = score
        return alpha

    def alpha_beta(self, board, depth, alpha, beta, can_null=True):
        key = chess.polyglot.zobrist_hash(board)
        hash_move = None
        
        if key in self.tt:
            d, val, m = self.tt[key]
            if d >= depth: return val
            hash_move = m

        if depth <= 0:
            return self.quiescence(board, alpha, beta)

        # --- Null Move Pruning ---
        if can_null and depth >= 3 and not board.is_check():
            board.push(chess.Move.null())
            score = -self.alpha_beta(board, depth - 3, -beta, -beta + 1, False)
            board.pop()
            if score >= beta: return beta

        moves = self.order_moves(board, depth, hash_move)
        if not moves:
            return -INF + 64 if board.is_check() else 0

        best_val = -INF
        best_move = moves[0]

        for i, move in enumerate(moves):
            board.push(move)
            
            # Late Move Reduction (LMR) - Cực mạnh để tăng tốc
            if i > 4 and depth >= 3 and not board.is_capture(move) and not board.is_check():
                score = -self.alpha_beta(board, depth - 2, -alpha - 1, -alpha)
                if score > alpha:
                    score = -self.alpha_beta(board, depth - 1, -beta, -alpha)
            else:
                score = -self.alpha_beta(board, depth - 1, -beta, -alpha)
            
            board.pop()

            if score >= beta:
                if not board.is_capture(move):
                    self.killers[depth][1] = self.killers[depth][0]
                    self.killers[depth][0] = move
                    self.history[move.from_square][move.to_square] += depth * depth
                return beta
            
            if score > best_val:
                best_val = score
                best_move = move
                if score > alpha: alpha = score

        self.tt[key] = (depth, best_val, best_move)
        return best_val

    def get_best_move(self, board, depth=3):
        best_move = None
        # --- Iterative Deepening ---
        # Tìm từ depth 1 lên đến depth mục tiêu
        for d in range(1, depth + 1):
            alpha, beta = -INF, INF
            moves = self.order_moves(board, d)
            if not moves: break
            
            current_best = moves[0]
            for move in moves:
                board.push(move)
                score = -self.alpha_beta(board, d - 1, -beta, -alpha)
                board.pop()
                if score > alpha:
                    alpha = score
                    current_best = move
            best_move = current_best
            
        return best_move