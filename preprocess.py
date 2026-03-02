"""
preprocess.py — Tiền xử lý dữ liệu PGN cho Chess Bot
═══════════════════════════════════════════════════════
Cải tiến:
  1. Label đúng perspective
  2. 14 kênh input (12 quân + lượt đi + nhập thành)
  3. Train/val split tự động
  4. Multiprocessing: xử lý tensor song song trên nhiều CPU core
"""

import chess
import chess.pgn
import numpy as np
import torch
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count


# ──────────────────────────────────────────────
# CẤU HÌNH
# ──────────────────────────────────────────────
CONFIG = {
    "pgn_path":    "lichess_db_standard_rated_2015-07.pgn",
    "output_path": "chess-data.pt",
    "max_games":   15000,   # 5k–10k là đủ cho bài tập
    "min_elo":     1800,    # Hạ xuống 1800 để lọc được nhiều ván hơn, nhanh hơn
    "skip_plies":  10,
    "val_ratio":   0.1,
    "num_workers": max(1, cpu_count() - 8),  # Dùng tất cả core trừ 1
    "chunk_size":  500,     # Số ván xử lý mỗi batch multiprocessing
}


# ──────────────────────────────────────────────
# BOARD → TENSOR (14 kênh)
# ──────────────────────────────────────────────
PIECE_MAP = {
    chess.PAWN:   0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK:   3,
    chess.QUEEN:  4,
    chess.KING:   5,
}

def board_to_tensor(board: chess.Board) -> np.ndarray:
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
    return tensor

def get_label(result: str, turn: chess.Color) -> float:
    white_wins = (result == "1-0")
    if turn == chess.WHITE:
        return 1.0 if white_wins else -1.0
    else:
        return -1.0 if white_wins else 1.0


# ──────────────────────────────────────────────
# XỬ LÝ 1 VÁN ĐẤU → list (tensor, label)
# Hàm này chạy trong subprocess (phải ở top-level để pickle được)
# ──────────────────────────────────────────────
def process_game(args) -> list:
    """Nhận (moves_uci, result, skip_plies), trả về list (tensor_14x8x8, label)."""
    moves_uci, result, skip_plies = args
    board = chess.Board()
    samples = []
    for i, uci in enumerate(moves_uci):
        move = chess.Move.from_uci(uci)
        board.push(move)
        if i >= skip_plies:
            samples.append((
                board_to_tensor(board),
                get_label(result, board.turn),
            ))
    return samples


# ──────────────────────────────────────────────
# ĐỌC PGN (1 luồng) → trả về list game_args
# ──────────────────────────────────────────────
def read_pgn_games(pgn_path: str, max_games: int, min_elo: int) -> list:
    """
    Đọc PGN và trích xuất thông tin cần thiết.
    Trả về list of (moves_uci, result, skip_plies).
    Chỉ truyền dữ liệu thuần (string/int) vào subprocess — không truyền object chess.
    """
    games = []
    skipped = 0
    skip_plies = CONFIG["skip_plies"]

    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        while len(games) < max_games:
            game = chess.pgn.read_game(f)
            if game is None:
                print("  ⚠ Đã đọc hết file PGN!")
                break

            result = game.headers.get("Result")
            if result not in ("1-0", "0-1"):
                skipped += 1
                continue

            try:
                white_elo = int(game.headers.get("WhiteElo", 0))
                black_elo = int(game.headers.get("BlackElo", 0))
            except ValueError:
                skipped += 1
                continue

            if white_elo < min_elo or black_elo < min_elo:
                skipped += 1
                continue

            # Chỉ lưu danh sách UCI string (nhẹ, dễ serialize)
            moves_uci = [m.uci() for m in game.mainline_moves()]
            games.append((moves_uci, result, skip_plies))

            if len(games) % 1000 == 0:
                print(f"  Đã đọc {len(games):,}/{max_games:,} ván "
                      f"(bỏ qua {skipped:,})...")

    return games


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def create_dataset(
    pgn_path:    str   = CONFIG["pgn_path"],
    output_path: str   = CONFIG["output_path"],
    max_games:   int   = CONFIG["max_games"],
    min_elo:     int   = CONFIG["min_elo"],
    val_ratio:   float = CONFIG["val_ratio"],
    num_workers: int   = CONFIG["num_workers"],
):
    print("╔" + "═"*52 + "╗")
    print("║        CHESS DATA PREPROCESSING               ║")
    print("╚" + "═"*52 + "╝")
    print(f"  File PGN   : {pgn_path}")
    print(f"  Max ván    : {max_games:,}")
    print(f"  Min Elo    : {min_elo}")
    print(f"  CPU workers: {num_workers} / {cpu_count()} cores")
    print()

    if not Path(pgn_path).exists():
        raise FileNotFoundError(f"Không tìm thấy file PGN: {pgn_path}")

    total_start = time.time()

    # ── Bước 1: Đọc PGN (single-thread) ──
    print("📖 Bước 1/3: Đọc và lọc PGN...")
    t0 = time.time()
    game_args = read_pgn_games(pgn_path, max_games, min_elo)
    print(f"  ✅ Đọc xong {len(game_args):,} ván ({time.time()-t0:.1f}s)")

    # ── Bước 2: Xử lý tensor song song ──
    print(f"\n⚙ Bước 2/3: Tạo tensor với {num_workers} workers...")
    t0 = time.time()

    all_inputs = []
    all_labels = []

    chunk = CONFIG["chunk_size"]
    with Pool(processes=num_workers) as pool:
        for i in range(0, len(game_args), chunk):
            batch = game_args[i:i+chunk]
            results = pool.map(process_game, batch)
            for samples in results:
                for tensor, label in samples:
                    all_inputs.append(tensor)
                    all_labels.append(label)

            done = min(i + chunk, len(game_args))
            pct  = done / len(game_args) * 100
            print(f"  [{done:>6}/{len(game_args)}] {pct:>5.1f}% | "
                  f"{len(all_inputs):,} thế trận | {time.time()-t0:.0f}s")

    print(f"  ✅ Xử lý xong {len(all_inputs):,} thế trận ({time.time()-t0:.1f}s)")

    # ── Bước 3: Lưu file ──
    print(f"\n💾 Bước 3/3: Shuffle & lưu file...")
    t0 = time.time()

    inputs_np = np.array(all_inputs, dtype=np.float32)
    labels_np = np.array(all_labels, dtype=np.float32)

    idx = np.random.permutation(len(inputs_np))
    inputs_np = inputs_np[idx]
    labels_np = labels_np[idx]

    split        = int(len(inputs_np) * (1 - val_ratio))
    train_inputs = torch.tensor(inputs_np[:split])
    train_labels = torch.tensor(labels_np[:split]).unsqueeze(1)
    val_inputs   = torch.tensor(inputs_np[split:])
    val_labels   = torch.tensor(labels_np[split:]).unsqueeze(1)

    torch.save({
        "train_inputs": train_inputs,
        "train_labels": train_labels,
        "val_inputs":   val_inputs,
        "val_labels":   val_labels,
        "meta": {
            "total_games":     len(game_args),
            "total_positions": len(all_inputs),
            "train_size":      len(train_inputs),
            "val_size":        len(val_inputs),
            "input_channels":  14,
            "min_elo":         min_elo,
        }
    }, output_path)

    total = time.time() - total_start
    print(f"  ✅ Lưu xong ({time.time()-t0:.1f}s)")
    print()
    print("╔" + "═"*52 + "╗")
    print("║  KẾT QUẢ                                      ║")
    print("╠" + "═"*52 + "╣")
    print(f"║  Tổng thế trận : {len(all_inputs):>10,}                  ║")
    print(f"║  Train         : {len(train_inputs):>10,}                  ║")
    print(f"║  Val           : {len(val_inputs):>10,}                  ║")
    print(f"║  Thời gian     : {total:>9.1f}s                  ║")
    print(f"║  File output   : {output_path:<34} ║")
    print("╚" + "═"*52 + "╝")


# ── Guard bắt buộc khi dùng multiprocessing trên Windows ──
if __name__ == "__main__":
    create_dataset()