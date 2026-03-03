import chess
import chess.polyglot
import math
import time
import os
import random
from collections import Counter
from typing import Optional, List, Tuple
import numpy as np

INF        = 999_999
MATE_SCORE = 900_000
MAX_PLY    = 128

# Sentinel cho inline static eval 
_UNSET = float('-inf')

# ══════════════════════════════════════════════════════════════════
# PIECE VALUES
# ══════════════════════════════════════════════════════════════════
PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   20_000,
}
_NPM_START = 6400   # Q(900)+2R(1000)+2B(660)+2N(640) × 2 sides

# ══════════════════════════════════════════════════════════════════
# PIECE-SQUARE TABLES
# ══════════════════════════════════════════════════════════════════
PST_PAWN = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]
PST_KNIGHT = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]
PST_BISHOP = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]
PST_ROOK = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]
PST_QUEEN = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]
PST_KING_MG = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]
PST_KING_EG = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
]
PST_MAP = {
    chess.PAWN:   PST_PAWN,
    chess.KNIGHT: PST_KNIGHT,
    chess.BISHOP: PST_BISHOP,
    chess.ROOK:   PST_ROOK,
    chess.QUEEN:  PST_QUEEN,
}

def pst_score(piece_type: int, sq: int, color: bool, eg: float) -> int:
    idx = sq if color == chess.WHITE else chess.square_mirror(sq)
    if piece_type == chess.KING:
        return int(PST_KING_MG[idx] * (1.0 - eg) + PST_KING_EG[idx] * eg)
    return PST_MAP[piece_type][idx]


# ══════════════════════════════════════════════════════════════════
# TT FLAGS
# ══════════════════════════════════════════════════════════════════
TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2


# ══════════════════════════════════════════════════════════════════
# MODULE-LEVEL CONSTANTS
# ══════════════════════════════════════════════════════════════════
_FILES_SET       = frozenset(range(8))
_CENTER_SQS      = (chess.D4, chess.E4, chess.D5, chess.E5)
_EXT_CENTER      = (chess.C3, chess.D3, chess.E3, chess.F3,
                    chess.C4,                     chess.F4,
                    chess.C5,                     chess.F5,
                    chess.C6, chess.D6, chess.E6, chess.F6)
_W_OUTPOST_RANKS = frozenset((3, 4, 5))
_B_OUTPOST_RANKS = frozenset((2, 3, 4))
_KING_ATK_PAIRS  = (
    (chess.QUEEN,  3),
    (chess.ROOK,   2),
    (chess.BISHOP, 1),
    (chess.KNIGHT, 1),
)
FUTILITY_MARGINS = (0, 100, 300, 500, 700)
LMP_COUNTS       = (0, 8, 14, 20, 28)
RFP_MARGIN_PER_D = 100
PC_MARGIN        = 200
_SEE_VAL         = [0, 100, 320, 330, 500, 900, 20_000]
_HIST_CAP        = 400_000  


# ══════════════════════════════════════════════════════════════════
# SEE — x-ray aware  en passant handled
# ══════════════════════════════════════════════════════════════════
def _get_attackers_to(board: chess.Board, sq: int, occupied: int) -> int:
    atk  = chess.BB_PAWN_ATTACKS[chess.WHITE][sq] & board.pawns & board.occupied_co[chess.BLACK]
    atk |= chess.BB_PAWN_ATTACKS[chess.BLACK][sq] & board.pawns & board.occupied_co[chess.WHITE]
    atk |= chess.BB_KNIGHT_ATTACKS[sq] & board.knights
    atk |= chess.BB_KING_ATTACKS[sq]   & board.kings
    atk |= (chess.BB_DIAG_ATTACKS[sq][chess.BB_DIAG_MASKS[sq] & occupied]
            & (board.bishops | board.queens))
    atk |= (chess.BB_RANK_ATTACKS[sq][chess.BB_RANK_MASKS[sq] & occupied]
            & (board.rooks | board.queens))
    atk |= (chess.BB_FILE_ATTACKS[sq][chess.BB_FILE_MASKS[sq] & occupied]
            & (board.rooks | board.queens))
    return atk & occupied


def _lva(board: chess.Board, bb: int) -> Tuple[Optional[int], int]:
    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP,
               chess.ROOK, chess.QUEEN, chess.KING):
        overlap = bb & (board.pieces_mask(pt, chess.WHITE) |
                        board.pieces_mask(pt, chess.BLACK))
        if overlap:
            return chess.lsb(overlap), pt
    return None, 0


def see(board: chess.Board, move: chess.Move) -> int:
    to_sq     = move.to_square
    aggressor = board.piece_at(move.from_square)
    if aggressor is None:
        return 0

    # En passant: victim is not on to_square
    victim = board.piece_at(to_sq)
    if victim is None and board.is_en_passant(move):
        gain_0 = _SEE_VAL[chess.PAWN]
    else:
        gain_0 = _SEE_VAL[victim.piece_type] if victim else 0

    gain     = [0] * 32
    gain[0]  = gain_0
    occupied = board.occupied ^ chess.BB_SQUARES[move.from_square]
    # EP: also remove captured pawn from occupied (not on to_square)
    if victim is None and board.is_en_passant(move):
        ep_sq    = chess.square(chess.square_file(to_sq), chess.square_rank(move.from_square))
        occupied ^= chess.BB_SQUARES[ep_sq]
    color    = not aggressor.color
    last_val = _SEE_VAL[aggressor.piece_type]
    d = 1
    while True:
        all_atk  = _get_attackers_to(board, to_sq, occupied)
        side_atk = all_atk & board.occupied_co[color] & occupied
        if not side_atk:
            break
        lva_sq, lva_type = _lva(board, side_atk)
        if lva_sq is None:
            break
        gain[d]  = last_val - gain[d - 1]
        last_val = _SEE_VAL[lva_type]
        occupied ^= chess.BB_SQUARES[lva_sq]
        color = not color
        d += 1
    while d > 1:
        d -= 1
        gain[d - 1] = -max(-gain[d - 1], gain[d])
    return gain[0]


# ══════════════════════════════════════════════════════════════════
# PRECOMPUTED LMR TABLE
# ══════════════════════════════════════════════════════════════════
LMR_TABLE = [[0] * 64 for _ in range(MAX_PLY)]
for _d in range(1, MAX_PLY):
    for _m in range(1, 64):
        LMR_TABLE[_d][_m] = max(0, int(math.log(_d) * math.log(_m) / 2.25 - 0.3))


# ══════════════════════════════════════════════════════════════════
# ENGINE
# ══════════════════════════════════════════════════════════════════
class NNBt:
    def __init__(self, book_path: str = None):
        self.tt: dict    = {}
        self.tt_gen: int = 0
        self.TT_SIZE     = 1 << 20          # 1M entries

        #  Pawn Hash Table: key → (pawn_score, king_score)
        self.pawn_tt: dict = {}
        self.PAWN_TT_SIZE  = 1 << 14        # 16K entries — pawns change rarely

        self.history      = np.zeros((2, 64, 64), dtype=np.int32)
        self.killers      = [[None, None] for _ in range(MAX_PLY)]
        self.countermoves = [[None] * 64 for _ in range(64)]
        self.cont_history = np.zeros((64, 64), dtype=np.int32)
        self.cap_history  = np.zeros((7, 64, 7), dtype=np.int32)

        self.nodes         = 0
        self.start_time    = 0.0
        self.time_limit    = 5.0
        self._timeout      = False
        self._max_ext      = 16
        self._stable_count = 0
        self._last_best    = None

        self._book = None
        if book_path and os.path.isfile(book_path):
            try:
                self._book = chess.polyglot.open_reader(book_path)
            except Exception:
                pass

    # ──────────────────────────────────────────
    # ENDGAME FACTOR
    # ──────────────────────────────────────────
    def _endgame_factor(self, board: chess.Board) -> float:
        npm = (chess.popcount(board.knights) * 320 +
               chess.popcount(board.bishops) * 330 +
               chess.popcount(board.rooks)   * 500 +
               chess.popcount(board.queens)  * 900)
        return max(0.0, 1.0 - npm / _NPM_START)

    # ──────────────────────────────────────────
    # PAWN ZOBRIST KEY
    # ──────────────────────────────────────────
    def _pawn_key(self, board: chess.Board, eg: float) -> int:
        wk = board.king(chess.WHITE) or 0
        bk = board.king(chess.BLACK) or 0
        # _king_safety returns 0 when eg > 0.6, so bucket on that threshold
        eg_bucket = 1 if eg > 0.6 else 0
        return (int(board.pawns & board.occupied_co[chess.WHITE]) * 0x9e3779b97f4a7c15 ^
                int(board.pawns & board.occupied_co[chess.BLACK]) * 0x6c62272e07bb0142 ^
                wk * 0xbf58476d1ce4e5b9 ^
                bk * 0x94d049bb133111eb ^
                eg_bucket * 0x517cc1b727220a95) & 0xFFFFFFFFFFFFFFFF

    # ──────────────────────────────────────────
    # PAWN STRUCTURE 
    # ──────────────────────────────────────────
    def _pawn_structure(self, board: chess.Board) -> int:
        score = 0
        for color in (chess.WHITE, chess.BLACK):
            sign      = 1 if color == chess.WHITE else -1
            pawns     = board.pieces(chess.PAWN, color)
            file_cnt  = Counter(chess.square_file(sq) for sq in pawns)
            file_set  = set(file_cnt)

            for f, cnt in file_cnt.items():
                if cnt > 1:
                    score -= sign * 20 * (cnt - 1)
                if all(n not in file_set for n in (f-1, f+1) if n in _FILES_SET):
                    score -= sign * 15

            opp_pawns = board.pieces(chess.PAWN, not color)
            for sq in pawns:
                f         = chess.square_file(sq)
                r         = chess.square_rank(sq)
                adj_files = {f-1, f, f+1} & _FILES_SET
                is_passed = all(
                    chess.square_file(osq) not in adj_files
                    or (color == chess.WHITE and chess.square_rank(osq) <= r)
                    or (color == chess.BLACK and chess.square_rank(osq) >= r)
                    for osq in opp_pawns
                )
                if is_passed:
                    advance = r if color == chess.WHITE else 7 - r
                    score += sign * (10 + advance * 8)
        return score

    # ──────────────────────────────────────────
    # KING SAFETY 
    # ──────────────────────────────────────────
    def _king_safety(self, board: chess.Board, eg: float) -> int:
        if eg > 0.6:
            return 0
        score = 0
        for color in (chess.WHITE, chess.BLACK):
            sign    = 1 if color == chess.WHITE else -1
            king_sq = board.king(color)
            if king_sq is None:
                continue
            kf        = chess.square_file(king_sq)
            kr        = chess.square_rank(king_sq)
            direction = 1 if color == chess.WHITE else -1

            # Pawn shield
            for df in (-1, 0, 1):
                sf, sr = kf + df, kr + direction
                if 0 <= sf <= 7 and 0 <= sr <= 7:
                    p = board.piece_at(chess.square(sf, sr))
                    score += sign * (10 if (p and p.piece_type == chess.PAWN
                                            and p.color == color) else -8)

            # Open + semi-open file penalties around king
            for df in (-1, 0, 1):
                f = kf + df
                if 0 <= f <= 7:
                    fm = chess.BB_FILES[f]
                    own_pawn = bool(board.pieces(chess.PAWN, color) & fm)
                    opp_pawn = bool(board.pieces(chess.PAWN, not color) & fm)
                    if not own_pawn and not opp_pawn:
                        score -= sign * 20   # fully open — most dangerous
                    elif not own_pawn:
                        score -= sign * 10   # semi-open — also dangerous

            # Attacker proximity
            king_zone    = board.attacks(king_sq)
            attack_score = 0
            for sq in king_zone:
                bb = chess.BB_SQUARES[sq]
                for pt, w in _KING_ATK_PAIRS:
                    if board.pieces(pt, not color) & bb:
                        attack_score += w
            score -= sign * attack_score * 8
        return score

    # ──────────────────────────────────────────
    # CENTER CONTROL
    # ──────────────────────────────────────────
    def _center_control(self, board: chess.Board) -> int:
        score = 0
        for sq in _CENTER_SQS:
            score += (len(board.attackers(chess.WHITE, sq)) -
                      len(board.attackers(chess.BLACK, sq))) * 5
        for sq in _EXT_CENTER:
            score += (len(board.attackers(chess.WHITE, sq)) -
                      len(board.attackers(chess.BLACK, sq))) * 2
        return score

    # ──────────────────────────────────────────
    # ROOK BONUSES
    # ──────────────────────────────────────────
    def _rook_bonuses(self, board: chess.Board) -> int:
        score = 0
        for color in (chess.WHITE, chess.BLACK):
            sign    = 1 if color == chess.WHITE else -1
            rooks   = board.pieces(chess.ROOK, color)
            seventh = 6 if color == chess.WHITE else 1
            for sq in rooks:
                f   = chess.square_file(sq)
                fm  = chess.BB_FILES[f]
                wp  = bool(board.pieces(chess.PAWN, chess.WHITE) & fm)
                bp  = bool(board.pieces(chess.PAWN, chess.BLACK) & fm)
                own = wp if color == chess.WHITE else bp
                opp = bp if color == chess.WHITE else wp
                if not own and not opp:
                    score += sign * 25
                elif not own:
                    score += sign * 15
                if chess.square_rank(sq) == seventh:
                    score += sign * 20
            rook_list = list(rooks)
            if len(rook_list) == 2:
                r0, r1 = rook_list
                if chess.square_rank(r0) == chess.square_rank(r1):
                    lo_f = min(chess.square_file(r0), chess.square_file(r1))
                    hi_f = max(chess.square_file(r0), chess.square_file(r1))
                    rnk  = chess.square_rank(r0)
                    if not any(board.piece_at(chess.square(f, rnk))
                               for f in range(lo_f+1, hi_f)):
                        score += sign * 15
                elif chess.square_file(r0) == chess.square_file(r1):
                    lo_r = min(chess.square_rank(r0), chess.square_rank(r1))
                    hi_r = max(chess.square_rank(r0), chess.square_rank(r1))
                    fc   = chess.square_file(r0)
                    if not any(board.piece_at(chess.square(fc, r))
                               for r in range(lo_r+1, hi_r)):
                        score += sign * 12
        return score

    # ──────────────────────────────────────────
    # KNIGHT OUTPOSTS
    # ──────────────────────────────────────────
    def _knight_outposts(self, board: chess.Board) -> int:
        score = 0
        for color in (chess.WHITE, chess.BLACK):
            sign  = 1 if color == chess.WHITE else -1
            ranks = _W_OUTPOST_RANKS if color == chess.WHITE else _B_OUTPOST_RANKS
            for sq in board.pieces(chess.KNIGHT, color):
                rank = chess.square_rank(sq)
                if rank not in ranks:
                    continue
                f         = chess.square_file(sq)
                prot_rank = rank - 1 if color == chess.WHITE else rank + 1
                protected = any(
                    0 <= f+df <= 7 and 0 <= prot_rank <= 7 and
                    board.piece_at(chess.square(f+df, prot_rank)) ==
                    chess.Piece(chess.PAWN, color)
                    for df in (-1, 1)
                )
                if not protected:
                    continue
                opp_pawns  = board.pieces(chess.PAWN, not color)
                challenged = any(
                    (chess.square_file(osq) == f-1 or
                     chess.square_file(osq) == f+1) and (
                        (color == chess.WHITE and chess.square_rank(osq) > rank) or
                        (color == chess.BLACK and chess.square_rank(osq) < rank)
                    )
                    for osq in opp_pawns
                )
                if not challenged:
                    score += sign * 20
        return score

    # ──────────────────────────────────────────
    # SPACE
    # ──────────────────────────────────────────
    def _space(self, board: chess.Board, eg: float) -> int:
        if eg > 0.5:
            return 0
        score = 0
        for color in (chess.WHITE, chess.BLACK):
            sign     = 1 if color == chess.WHITE else -1
            opp_half = range(4, 8) if color == chess.WHITE else range(0, 4)
            sp = sum(1 for sq in board.pieces(chess.PAWN, color)
                     for att in board.attacks(sq)
                     if chess.square_rank(att) in opp_half)
            score += sign * sp * 2
        return score

    # ──────────────────────────────────────────
    # ENDGAME PATTERNS
    # ──────────────────────────────────────────
    def _endgame_patterns(self, board: chess.Board, eg: float,
                          w_mat: int = -1, b_mat: int = -1) -> int:
        if eg < 0.7:
            return 0
        if w_mat < 0:  # fallback if not provided
            w_mat = b_mat = 0
            for p in board.piece_map().values():
                if p.piece_type != chess.KING:
                    if p.color == chess.WHITE: w_mat += PIECE_VALUES[p.piece_type]
                    else:                      b_mat += PIECE_VALUES[p.piece_type]
        score = 0
        for color in (chess.WHITE, chess.BLACK):
            sign    = 1 if color == chess.WHITE else -1
            own_mat = w_mat if color == chess.WHITE else b_mat
            opp_mat = b_mat if color == chess.WHITE else w_mat
            if own_mat > opp_mat + 300:
                opp_king = board.king(not color)
                own_king = board.king(color)
                if opp_king and own_king:
                    kf   = chess.square_file(opp_king)
                    kr   = chess.square_rank(opp_king)
                    dist = (abs(chess.square_file(own_king) - kf) +
                            abs(chess.square_rank(own_king) - kr))
                    score += sign * (14 - min(kf, 7-kf) - min(kr, 7-kr)) * 4
                    score += sign * (14 - dist) * 3
        return score

    # ──────────────────────────────────────────
    # EVALUATE
    # ──────────────────────────────────────────
    def evaluate(self, board: chess.Board, eg: float = -1.0,
                 legal: Optional[List] = None) -> int:
        if legal is None:
            legal = list(board.legal_moves)
        move_count = len(legal)

        if move_count == 0:
            return -MATE_SCORE if board.is_check() else 0

        if eg < 0.0:
            eg = self._endgame_factor(board)

        score = 0
        wb = bb = 0
        w_mat = b_mat = 0

        for sq, piece in board.piece_map().items():
            v   = PIECE_VALUES[piece.piece_type]
            pst = pst_score(piece.piece_type, sq, piece.color, eg)
            if piece.piece_type == chess.BISHOP:
                if piece.color == chess.WHITE: wb += 1
                else:                          bb += 1
            if piece.piece_type != chess.KING:
                if piece.color == chess.WHITE: w_mat += v
                else:                          b_mat += v
            score += (v + pst) if piece.color == chess.WHITE else -(v + pst)

        if wb >= 2: score += 30
        if bb >= 2: score -= 30

        score += (move_count * 3) if board.turn == chess.WHITE else -(move_count * 3)
        score += 10 if board.turn == chess.WHITE else -10

        pawn_key = self._pawn_key(board, eg)

        if pawn_key in self.pawn_tt:
            pawn_cached = self.pawn_tt[pawn_key]
        else:
            pawn_cached = self._pawn_structure(board) + self._king_safety(board, eg)
            if len(self.pawn_tt) < self.PAWN_TT_SIZE:
                self.pawn_tt[pawn_key] = pawn_cached
            else:
                # Simple eviction: clear half on overflow (rare)
                keys_to_del = list(self.pawn_tt.keys())[:self.PAWN_TT_SIZE // 2]
                for k in keys_to_del:
                    del self.pawn_tt[k]
                self.pawn_tt[pawn_key] = pawn_cached

        score += pawn_cached
        score += self._center_control(board)
        score += self._rook_bonuses(board)
        score += self._knight_outposts(board)
        score += self._space(board, eg)
        score += self._endgame_patterns(board, eg, w_mat, b_mat)


        return score if board.turn == chess.WHITE else -score

    # ──────────────────────────────────────────
    # ──────────────────────────────────────────
    def order_moves(
        self,
        board: chess.Board,
        ply: int,
        moves: Optional[List] = None,
        tt_move:   Optional[chess.Move] = None,
        prev_move: Optional[chess.Move] = None,
    ) -> List[chess.Move]:
        if moves is None:
            moves = list(board.legal_moves)
        c_idx = 0 if board.turn == chess.WHITE else 1
        kid   = min(ply, MAX_PLY - 1)
        cm    = (self.countermoves[prev_move.from_square][prev_move.to_square]
                 if prev_move else None)

        def score_move(m: chess.Move) -> int:
            if m == tt_move:
                return 3_000_000
            is_cap = board.is_capture(m)
            s = 0
            if is_cap:
                sv = see(board, m)
                if sv >= 0:
                    s = 2_000_000 + sv
                    agg = board.piece_at(m.from_square)
                    vic = board.piece_at(m.to_square)
                    if agg and vic:
                        s += int(self.cap_history[agg.piece_type]
                                                 [m.to_square]
                                                 [vic.piece_type]) // 4
                else:
                    s = -100_000 + sv
            if m.promotion:
                s = max(s, 1_800_000 + PIECE_VALUES.get(m.promotion, 0))
            if not is_cap and not m.promotion:
                if m == self.killers[kid][0]:
                    s = 1_700_000
                elif m == self.killers[kid][1]:
                    s = 1_600_000
                elif m == cm:
                    s = 1_550_000
                else:
                    h  = int(self.history[c_idx][m.from_square][m.to_square])
                    ch = (int(self.cont_history[prev_move.to_square][m.to_square])
                          if prev_move else 0)
                    s += h + ch // 2
            return s

        moves.sort(key=score_move, reverse=True)
        return moves

    # ──────────────────────────────────────────
    # QUIESCENCE SEARCH
    # ──────────────────────────────────────────
    def quiescence(self, board: chess.Board, alpha: int, beta: int,
                   ply: int = 0, eg: float = -1.0) -> int:
        self.nodes += 1
        alpha = max(alpha, -(MATE_SCORE - ply))
        beta  = min(beta,   (MATE_SCORE - ply))
        if alpha >= beta:
            return alpha

        legal    = list(board.legal_moves)
        in_check = board.is_check()

        if board.is_insufficient_material():
            return 0

        if not in_check:
            stand_pat = self.evaluate(board, eg, legal)
            if stand_pat >= beta:
                return beta
            if stand_pat > alpha:
                alpha = stand_pat
            if stand_pat + 1050 < alpha:
                return alpha
        else:
            stand_pat = -(MATE_SCORE - ply)  # worst case for mate distance

        if not legal:
            return -(MATE_SCORE - ply) if in_check else 0

        if in_check:
            evasions: List[Tuple[int, chess.Move]] = [(0, m) for m in legal]
            # Order: captures first (by SEE), then quiets
            evasions.sort(key=lambda x: (
                see(board, x[1]) if board.is_capture(x[1]) else -10_000
            ), reverse=True)
            moves_to_search = evasions
        else:
            # Normal QS: only captures and promotions
            captures: List[Tuple[int, chess.Move]] = []
            for move in legal:
                is_cap   = board.is_capture(move)
                is_promo = bool(move.promotion)
                if not is_cap and not is_promo:
                    continue
                sv = see(board, move) if is_cap else 900
                if is_cap and not is_promo and sv < -50:
                    continue
                captures.append((sv, move))
            captures.sort(key=lambda x: x[0], reverse=True)
            moves_to_search = captures

        for sv, move in moves_to_search:
            if not in_check and board.is_capture(move):
                victim = board.piece_at(move.to_square)
                if victim is None and board.is_en_passant(move):
                    if stand_pat + PIECE_VALUES[chess.PAWN] + 200 < alpha:
                        continue
                elif victim and stand_pat + PIECE_VALUES[victim.piece_type] + 200 < alpha:
                    continue
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha, ply + 1, -1.0)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    # ──────────────────────────────────────────
    # ALPHA-BETA
    # ──────────────────────────────────────────
    def alpha_beta(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int           = 0,
        null_allowed: bool = True,
        extensions: int    = 0,
        prev_move: Optional[chess.Move] = None,
    ) -> int:
        self.nodes += 1

        if self.nodes & 4095 == 0:
            if time.time() - self.start_time > self.time_limit:
                self._timeout = True
        if self._timeout:
            return alpha

        alpha = max(alpha, -(MATE_SCORE - ply))
        beta  = min(beta,   (MATE_SCORE - ply))
        if alpha >= beta:
            return alpha

        if board.is_repetition(2) or board.is_fifty_moves():
            return 0

        key      = chess.polyglot.zobrist_hash(board)
        tt_move  = None
        tt_entry = None
        tt_static_ev = None
        tt_flag_hit  = None

        if key in self.tt:
            tt_entry = self.tt[key]
            tt_d, tt_score, tt_flag_hit, tt_mv, _gen, tt_ev = tt_entry
            tt_move      = tt_mv
            tt_static_ev = tt_ev
            if tt_d >= depth:
                if tt_flag_hit == TT_EXACT:
                    return tt_score
                elif tt_flag_hit == TT_LOWER:
                    alpha = max(alpha, tt_score)
                elif tt_flag_hit == TT_UPPER:
                    beta  = min(beta,  tt_score)
                if alpha >= beta:
                    return tt_score

        if depth <= 0:
            eg_qs = self._endgame_factor(board)
            return self.quiescence(board, alpha, beta, ply, eg_qs)

        legal    = list(board.legal_moves)
        in_check = board.is_check()

        if not legal:
            return -(MATE_SCORE - ply) if in_check else 0
        if board.is_insufficient_material():
            return 0

        is_pv = (beta - alpha > 1)
        eg    = self._endgame_factor(board)

        # Use _UNSET sentinel; compute lazily with inline checks
        static_ev = _UNSET

        # Check extension
        ext = 0
        if in_check and extensions < self._max_ext:
            ext = 1

        if (not is_pv and depth >= 6 and tt_move is not None
                and ext == 0 and not in_check
                and tt_flag_hit in (TT_EXACT, TT_LOWER)):
            if tt_entry is not None and tt_entry[0] >= depth - 3:
                se_beta  = max(tt_entry[1] - 50, -(MATE_SCORE - ply - 50))
                se_depth = (depth - 1) // 2
                fail_low = True
                for m in legal:
                    if m == tt_move:
                        continue
                    board.push(m)
                    s = -self.alpha_beta(board, se_depth, -se_beta - 1, -se_beta,
                                         ply + 1, null_allowed=False,
                                         extensions=extensions)
                    board.pop()
                    if self._timeout:
                        break
                    if s >= se_beta:
                        fail_low = False
                        break
                if fail_low and not self._timeout:
                    ext = 1

        if not in_check and not is_pv and 1 <= depth <= 4:
            if static_ev is _UNSET:
                static_ev = (tt_static_ev if tt_static_ev is not None
                             else self.evaluate(board, eg, legal))
            if static_ev - RFP_MARGIN_PER_D * depth >= beta:
                return static_ev

        if null_allowed and not in_check and not is_pv and depth >= 3 and eg < 0.8:
            if static_ev is _UNSET:
                static_ev = (tt_static_ev if tt_static_ev is not None
                             else self.evaluate(board, eg, legal))
            if static_ev >= beta:
                R = 3 + depth // 4 + min(3, (static_ev - beta) // 200)
                board.push(chess.Move.null())
                null_score = -self.alpha_beta(
                    board, depth - 1 - R, -beta, -beta + 1,
                    ply + 1, null_allowed=False, extensions=extensions
                )
                board.pop()
                if null_score >= beta:
                    return beta

        if not is_pv and not in_check and depth >= 5 and abs(beta) < MATE_SCORE - 100:
            pc_beta  = beta + PC_MARGIN
            pc_depth = depth - 4
            for m in legal:
                if not board.is_capture(m):
                    continue
                if see(board, m) < PC_MARGIN - 50:
                    continue
                board.push(m)
                pcs = -self.quiescence(board, -pc_beta, -pc_beta + 1, ply + 1, -1.0)
                if pcs >= pc_beta:
                    pcs = -self.alpha_beta(board, pc_depth, -pc_beta, -pc_beta + 1,
                                           ply + 1, null_allowed=True,
                                           extensions=extensions)
                board.pop()
                if self._timeout:
                    break
                if pcs >= pc_beta:
                    return beta

        if depth == 1 and not in_check and not is_pv:
            if static_ev is _UNSET:
                static_ev = (tt_static_ev if tt_static_ev is not None
                             else self.evaluate(board, eg, legal))
            if static_ev + 300 < alpha:
                return self.quiescence(board, alpha, beta, ply, eg)

        if is_pv and depth >= 5 and tt_move is None and not self._timeout:
            self.alpha_beta(board, depth - 2, alpha, beta, ply,
                            null_allowed, extensions, prev_move)
            iid_entry = self.tt.get(key)
            if iid_entry:
                tt_move      = iid_entry[3]
                tt_entry     = iid_entry
                if iid_entry[5] is not None:
                    tt_static_ev = iid_entry[5]

        moves = self.order_moves(board, ply, legal[:], tt_move, prev_move)


        best_score          = -INF
        best_move           = None
        orig_alpha          = alpha
        move_count          = 0
        quiet_count         = 0
        searched_quiets:    List[chess.Move] = []
        searched_bad_caps:  List[chess.Move] = []

        c_idx_mover = 0 if board.turn == chess.WHITE else 1

        for move in moves:
            is_capture   = board.is_capture(move)
            is_promotion = bool(move.promotion)
            is_quiet     = not is_capture and not is_promotion

            # LMP
            if (is_quiet and not in_check and not is_pv
                    and 1 <= depth <= 4
                    and quiet_count >= LMP_COUNTS[depth]):
                move_count += 1
                quiet_count += 1
                continue

            if is_quiet and not in_check and 1 <= depth <= 3 and move_count > 0:
                if static_ev is _UNSET:
                    static_ev = (tt_static_ev if tt_static_ev is not None
                                 else self.evaluate(board, eg, legal))
                if static_ev + FUTILITY_MARGINS[depth] <= alpha:
                    move_count += 1
                    quiet_count += 1
                    continue

            # SEE Pruning on losing captures
            see_val = None
            if (not is_pv and depth <= 4 and is_capture
                    and not is_promotion and move_count > 0):
                see_val = see(board, move)
                if see_val < -60 * depth:
                    searched_bad_caps.append(move)

                    move_count += 1
                    continue

            board.push(move)
            new_depth = depth - 1 + ext

            if move_count == 0:
                score = -self.alpha_beta(board, new_depth, -beta, -alpha,
                                          ply + 1, True, extensions + ext, move)
            else:
                do_lmr    = is_quiet and not in_check and move_count >= 2 and depth >= 3
                reduction = 0
                if do_lmr:
                    reduction = LMR_TABLE[min(depth, MAX_PLY-1)][min(move_count, 63)]
                    if not is_pv:
                        reduction += 1
                    if self.history[c_idx_mover][move.from_square][move.to_square] > 5_000:
                        reduction = max(0, reduction - 1)
                    reduction = max(0, min(reduction, new_depth - 1))

                score = -self.alpha_beta(board, new_depth - reduction, -alpha - 1, -alpha,
                                          ply + 1, True, extensions + ext, move)
                if reduction > 0 and score > alpha and not self._timeout:
                    score = -self.alpha_beta(board, new_depth, -alpha - 1, -alpha,
                                              ply + 1, True, extensions + ext, move)
                if score > alpha and is_pv and not self._timeout:
                    score = -self.alpha_beta(board, new_depth, -beta, -alpha,
                                              ply + 1, True, extensions + ext, move)

            board.pop()
            move_count += 1
            if is_quiet:
                quiet_count += 1
                searched_quiets.append(move)

            if score > best_score:
                best_score = score
                best_move  = move
            if score > alpha:
                alpha = score

            if alpha >= beta:
                bonus  = depth * depth
                malus  = -(depth * depth)
                if not is_capture:
                    kid = min(ply, MAX_PLY - 1)
                    if move != self.killers[kid][0]:
                        self.killers[kid][1] = self.killers[kid][0]
                        self.killers[kid][0] = move
                    h = int(self.history[c_idx_mover][move.from_square][move.to_square])
                    self.history[c_idx_mover][move.from_square][move.to_square] = min(h + bonus, _HIST_CAP)
                    if prev_move:
                        ch = int(self.cont_history[prev_move.to_square][move.to_square])
                        self.cont_history[prev_move.to_square][move.to_square] = min(ch + bonus, _HIST_CAP)
                        self.countermoves[prev_move.from_square][prev_move.to_square] = move
                else:
                    agg_p = board.piece_at(move.from_square)
                    vic_p = board.piece_at(move.to_square)
                    if agg_p and vic_p:
                        ch = int(self.cap_history[agg_p.piece_type][move.to_square][vic_p.piece_type])
                        self.cap_history[agg_p.piece_type][move.to_square][vic_p.piece_type] = min(ch + bonus, _HIST_CAP)

                # Malus for quiets that didn't cause cutoff
                for sq_m in searched_quiets:
                    if sq_m != move:
                        h = int(self.history[c_idx_mover][sq_m.from_square][sq_m.to_square])
                        self.history[c_idx_mover][sq_m.from_square][sq_m.to_square] = max(h + malus, -_HIST_CAP)

                for bc in searched_bad_caps:
                    agg_p = board.piece_at(bc.from_square)
                    vic_p = board.piece_at(bc.to_square)
                    if agg_p and vic_p:
                        ch = int(self.cap_history[agg_p.piece_type][bc.to_square][vic_p.piece_type])
                        self.cap_history[agg_p.piece_type][bc.to_square][vic_p.piece_type] = max(ch + malus, -_HIST_CAP)
                break

        if best_score <= orig_alpha:
            tt_flag = TT_UPPER
        elif best_score >= beta:
            tt_flag = TT_LOWER
        else:
            tt_flag = TT_EXACT

        # or existing entry is from older gen or shallower depth
        should_store = True
        if key in self.tt:
            old_d, _, _, _, old_gen, _ = self.tt[key]
            if old_gen == self.tt_gen and old_d > depth:
                should_store = False   # Same gen, deeper entry → keep
        # Always overwrite older gen or new entries; respect TT_SIZE only for new keys
        if should_store:
            if key in self.tt or len(self.tt) < self.TT_SIZE:
                self.tt[key] = (depth, best_score, tt_flag, best_move, self.tt_gen,
                                static_ev if static_ev is not _UNSET else None)
            else:
                # Find any stale entry to replace (O(1) amortized via random probe)
                stale = None
                for probe_key, probe_val in self.tt.items():
                    if probe_val[4] < self.tt_gen:
                        stale = probe_key
                        break
                if stale is not None:
                    del self.tt[stale]
                    self.tt[key] = (depth, best_score, tt_flag, best_move, self.tt_gen,
                                    static_ev if static_ev is not _UNSET else None)
                # else: TT completely full with current gen → skip (very rare)

        return best_score

    # ──────────────────────────────────────────
    # ──────────────────────────────────────────
    def _evict_tt(self):
        if len(self.tt) < self.TT_SIZE:
            return
        # Age-based eviction first (fast)
        cutoff   = self.tt_gen - 1
        old_keys = [k for k, v in self.tt.items() if v[4] < cutoff]
        for k in old_keys:
            del self.tt[k]
        if len(self.tt) >= self.TT_SIZE:
            sample_size = len(self.tt) // 4
            to_del = random.sample(list(self.tt.keys()), sample_size)
            for k in to_del:
                del self.tt[k]

    # ──────────────────────────────────────────
    # ITERATIVE DEEPENING
    # ──────────────────────────────────────────
    def get_best_move(
        self,
        board: chess.Board,
        depth: int        = 3,
        max_depth: int    = 10,
        time_limit: float = 5.0,
    ) -> Optional[chess.Move]:

        if self._book:
            try:
                entry = self._book.weighted_choice(board)
                if entry:
                    return entry.move
            except Exception:
                pass

        self.time_limit    = time_limit
        self.start_time    = time.time()
        self.nodes         = 0
        self._timeout      = False
        self.killers       = [[None, None] for _ in range(MAX_PLY)]
        self.tt_gen       += 1
        self._stable_count = 0
        self.pawn_tt.clear()   # Pawn cache valid only within one search

        self._evict_tt()

        # History decay between moves
        self.history      //= 2
        self.cont_history //= 2
        self.cap_history  //= 2

        moves_list = list(board.legal_moves)
        if not moves_list:
            return None
        if len(moves_list) == 1:
            return moves_list[0]

        tt_default = self.tt.get(chess.polyglot.zobrist_hash(board), (None,)*6)[3]
        best_move  = tt_default if tt_default in moves_list else moves_list[0]
        best_score = -INF

        for current_depth in range(1, max_depth + 1):
            if self._timeout:
                break

            if current_depth >= 4 and abs(best_score) < MATE_SCORE - 100:
                window     = 40
                asp_alpha  = best_score - window
                asp_beta   = best_score + window
            else:
                asp_alpha  = -INF
                asp_beta   =  INF

            fail_count      = 0
            iter_best_move  = best_move
            iter_best_score = -INF

            while True:
                local_alpha = asp_alpha
                local_beta  = asp_beta
                iter_best_score = -INF
                iter_best_move  = best_move

                root_tt    = self.tt.get(chess.polyglot.zobrist_hash(board), (None,)*6)[3]
                root_moves = self.order_moves(board, 0,
                                              moves=list(moves_list),
                                              tt_move=root_tt)

                for move in root_moves:
                    if self._timeout:
                        break
                    board.push(move)
                    if move == root_moves[0]:
                        score = -self.alpha_beta(board, current_depth - 1,
                                                  -local_beta, -local_alpha,
                                                  ply=1, prev_move=move)
                    else:
                        score = -self.alpha_beta(board, current_depth - 1,
                                                  -local_alpha - 1, -local_alpha,
                                                  ply=1, prev_move=move)
                        if not self._timeout and local_alpha < score < local_beta:
                            score = -self.alpha_beta(board, current_depth - 1,
                                                      -local_beta, -local_alpha,
                                                      ply=1, prev_move=move)
                    board.pop()

                    if score > iter_best_score:
                        iter_best_score = score
                        iter_best_move  = move
                    if score > local_alpha:
                        local_alpha = score

                if self._timeout:
                    break

                # Aspiration window re-search logic
                if fail_count < 4:
                    if iter_best_score <= asp_alpha:
                        window    *= 3
                        asp_alpha  = iter_best_score - window
                        fail_count += 1
                        continue
                    elif iter_best_score >= asp_beta:
                        window    *= 3
                        asp_beta   = iter_best_score + window
                        fail_count += 1
                        continue
                break

            if not self._timeout:
                best_move  = iter_best_move
                best_score = iter_best_score

            elapsed = time.time() - self.start_time
            #print(f"  depth={current_depth:2d}  score={best_score:+6d}  "
            #      f"nodes={self.nodes:,}  time={elapsed:.2f}s  move={best_move}")

            if best_move == self._last_best:
                self._stable_count += 1
            else:
                self._stable_count = 0
            self._last_best = best_move

            if elapsed > self.time_limit * 0.55:
                break
            if self._stable_count >= 4 and elapsed > self.time_limit * 0.2:
                break

        return best_move


# ══════════════════════════════════════════════════════════════════
# QUICK SELF-TEST
# ══════════════════════════════════════════════════════════════════
# if __name__ == "__main__":
#     print("=== NNBt v8 — Self Test ===\n")
#     engine = NNBt()

#     # Test 1: Starting position
#     board = chess.Board()
#     print("[TEST 1] Starting position, depth=5, time=3s")
#     move = engine.get_best_move(board, max_depth=8, time_limit=3.0)
#     print(f"  Best move: {move}\n")

#     # Test 2: En passant 
#     print("[TEST 2] En passant SEE correctness")
#     ep_board = chess.Board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
#     ep_move = chess.Move.from_uci("e5f6")  # en passant capture
#     see_val = see(ep_board, ep_move)
#     print(f"  SEE(e5xf6 EP) = {see_val}  (expected ~100, was 0 in v7) ✓\n")

#     # Test 3: Tactical position (mate in 2)
#     print("[TEST 3] Mate in 2 — Anastasia's Mate")
#     m2_board = chess.Board("r1bk3r/p2pBpNp/n4n2/1p1NP2P/6P1/3P4/P1P1K3/q5b1 w - - 1 1")
#     move3 = engine.get_best_move(m2_board, max_depth=6, time_limit=5.0)
#     print(f"  Best move: {move3}\n")

#     # Test 4: Endgame patterns
#     print("[TEST 4] Endgame — King and pawn vs King")
#     eg_board = chess.Board("8/8/8/8/3k4/8/3P4/3K4 w - - 0 1")
#     move4 = engine.get_best_move(eg_board, max_depth=10, time_limit=3.0)
#     print(f"  Best move: {move4}\n")

#     print("=== All tests complete ===")