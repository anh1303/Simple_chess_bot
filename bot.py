"""
bot.py — Chess Bot với Neural Network + Alpha-Beta Minimax + Quiescence Search
═══════════════════════════════════════════════════════════════════════════════
Pipeline tìm nước đi:

  get_best_move()
      └─ Iterative Deepening (depth 1 → N)
            └─ alpha_beta()  ← Minimax chính (giữ nguyên yêu cầu)
                  ├─ depth > 0 : đệ quy Minimax bình thường
                  └─ depth = 0 : gọi quiescence() thay vì evaluate() thẳng
                                      └─ search tiếp các nước ăn quân
                                            └─ "yên tĩnh" → evaluate()

Cải tiến so với phiên bản trước:
  [MỚI] Quiescence search — tránh evaluate sai khi thế trận đang "nổ"
  [MỚI] Delta pruning   — cắt tỉa sớm trong quiescence nếu vô vọng
  [GIỮ] Minimax Alpha-Beta (Negamax)
  [GIỮ] MVV-LVA move ordering
  [GIỮ] Transposition table
  [GIỮ] Iterative deepening
  [GIỮ] 14-kênh input (match preprocess.py + train.py)
"""

import chess
import torch
import numpy as np
from train import ChessNet


# ──────────────────────────────────────────────
# HẰNG SỐ
# ──────────────────────────────────────────────
INF = float("inf")

PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   20000,
}

PIECE_MAP = {
    chess.PAWN:   0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK:   3,
    chess.QUEEN:  4,
    chess.KING:   5,
}

# Delta pruning: nếu ngay cả ăn quân giá trị cao nhất vẫn thua xa thì bỏ qua
DELTA_MARGIN = 0.3   # Đơn vị giống score NN: [-1, 1]


# ──────────────────────────────────────────────
# BOT
# ──────────────────────────────────────────────
class NNBt:
    def __init__(self, model_path: str, use_tt: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = ChessNet().to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

        self.use_tt = use_tt
        self.tt: dict = {}       # Transposition table: zobrist → (depth, score)
        self.nodes_searched = 0
        self.q_nodes        = 0  # Đếm riêng nodes trong quiescence

    # ────────────────────────────────────────
    # Board → Tensor (14 kênh)
    # ────────────────────────────────────────
    def board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        tensor = np.zeros((14, 8, 8), dtype=np.float32)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                offset = 0 if piece.color == chess.WHITE else 6
                layer  = PIECE_MAP[piece.piece_type] + offset
                row, col = divmod(square, 8)
                tensor[layer, row, col] = 1.0
        if board.turn == chess.WHITE:
            tensor[12, :, :] = 1.0
        if board.has_castling_rights(chess.WHITE):
            tensor[13, 0, :] = 1.0
        if board.has_castling_rights(chess.BLACK):
            tensor[13, 7, :] = 1.0
        return torch.from_numpy(tensor).unsqueeze(0).to(self.device)

    # ────────────────────────────────────────
    # Evaluate (NN) — góc nhìn người đang đến lượt
    # ────────────────────────────────────────
    def evaluate(self, board: chess.Board) -> float:
        if board.is_checkmate():
            return -9999.0
        if board.is_game_over():
            return 0.0
        with torch.no_grad():
            return self.model(self.board_to_tensor(board)).item()

    # ────────────────────────────────────────
    # Move Ordering — MVV-LVA + phong cấp
    # ────────────────────────────────────────
    def move_score(self, board: chess.Board, move: chess.Move) -> int:
        score = 0
        if board.is_capture(move):
            victim   = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            if victim and attacker:
                score += PIECE_VALUES.get(victim.piece_type, 0) * 10 \
                       - PIECE_VALUES.get(attacker.piece_type, 0)
        if move.promotion:
            score += PIECE_VALUES.get(move.promotion, 0)
        return score

    def order_moves(self, board: chess.Board, captures_only: bool = False):
        moves = [m for m in board.legal_moves
                 if not captures_only or board.is_capture(m) or m.promotion]
        return sorted(moves, key=lambda m: self.move_score(board, m), reverse=True)

    # ────────────────────────────────────────
    # QUIESCENCE SEARCH
    # Chỉ được gọi từ alpha_beta() khi depth=0.
    # Tiếp tục search các nước ăn quân cho đến khi thế trận "yên tĩnh",
    # sau đó mới evaluate() — tránh đánh giá sai thế trận đang bị ăn quân.
    # ────────────────────────────────────────
    def quiescence(self, board: chess.Board, alpha: float, beta: float) -> float:
        self.q_nodes += 1

        # "Stand pat": điểm hiện tại nếu không làm gì thêm
        # Nếu stand_pat đã >= beta thì đối thủ sẽ tránh nước này → cắt tỉa
        stand_pat = self.evaluate(board)
        if stand_pat >= beta:
            return beta
        
        # Delta pruning: nếu ngay cả ăn quân Hậu cũng không vượt alpha
        # thì không cần search thêm (thế trận quá tệ, vô vọng)
        if stand_pat < alpha - DELTA_MARGIN:
            return alpha

        alpha = max(alpha, stand_pat)

        # Chỉ xét nước ăn quân và phong cấp (không xét nước bình thường)
        for move in self.order_moves(board, captures_only=True):
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha)
            board.pop()

            if score >= beta:
                return beta      # Beta cutoff
            alpha = max(alpha, score)

        # Thế trận "yên tĩnh" — không còn nước ăn quân có lợi
        return alpha

    # ────────────────────────────────────────
    # MINIMAX ALPHA-BETA (Negamax)
    # Đây là thuật toán Minimax chính, giữ nguyên theo yêu cầu.
    # Thay đổi duy nhất: khi depth=0 gọi quiescence() thay vì evaluate().
    # ────────────────────────────────────────
    def alpha_beta(self, board: chess.Board, depth: int,
                   alpha: float, beta: float) -> float:
        self.nodes_searched += 1

        # Transposition table lookup
        key = board.zobrist_hash() if self.use_tt else None
        if key and key in self.tt:
            cached_depth, cached_score = self.tt[key]
            if cached_depth >= depth:
                return cached_score

        # Game over
        if board.is_game_over():
            return self.evaluate(board)

        # ── depth=0: vào quiescence thay vì evaluate thẳng ──
        if depth == 0:
            return self.quiescence(board, alpha, beta)

        # ── depth>0: Minimax bình thường ──
        best = -INF
        for move in self.order_moves(board):
            board.push(move)
            score = -self.alpha_beta(board, depth - 1, -beta, -alpha)
            board.pop()

            if score > best:
                best = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break   # Beta cutoff

        if key:
            self.tt[key] = (depth, best)

        return best

    # ────────────────────────────────────────
    # GET BEST MOVE — Iterative Deepening
    # ────────────────────────────────────────
    def get_best_move(self, board: chess.Board, depth: int = 2) -> chess.Move | None:
        legal = list(board.legal_moves)
        if not legal:
            return None

        self.nodes_searched = 0
        self.q_nodes        = 0
        best_move  = legal[0]

        # Iterative deepening: chạy từ depth=1 → depth
        # Kết quả depth thấp giúp order moves tốt hơn cho depth cao
        for current_depth in range(1, depth + 1):
            move_scores = []
            for move in self.order_moves(board):
                board.push(move)
                score = -self.alpha_beta(board, current_depth - 1, -INF, INF)
                board.pop()
                move_scores.append((score, move))

            move_scores.sort(key=lambda x: x[0], reverse=True)
            if move_scores:
                best_move = move_scores[0][1]

        return best_move

    def clear_tt(self):
        """Gọi đầu mỗi ván mới để tránh dùng cache từ ván cũ."""
        self.tt.clear()