"""
╔══════════════════════════════════════════════════════════════╗
║           CHESS BOT EVALUATION FRAMEWORK                     ║
║  Đánh giá toàn diện bot chess với nhiều cấp độ Stockfish     ║
╚══════════════════════════════════════════════════════════════╝
"""

import chess
import chess.engine
import time
import json
import os
from datetime import datetime
from dataclasses import dataclass, field, asdict
from bot_VIP import NNBt  # Thay bằng class bot của bạn

"""
"levels": [
    {"label": "Cực yếu",    "skill": 0},   → ~800 Elo
    {"label": "Rất yếu",    "skill": 1},   → ~900 Elo
    {"label": "Yếu",        "skill": 3},   → ~1050 Elo
    {"label": "Trung bình", "skill": 5},   → ~1200 Elo
    {"label": "Khá",        "skill": 8},   → ~1400 Elo
    {"label": "Mạnh",       "elo": 1500},  → 1500 Elo thật
    {"label": "Rất mạnh",   "elo": 1800},  → 1800 Elo thật
]
"""


# ──────────────────────────────────────────────
# CẤU HÌNH - Chỉnh tại đây
# ──────────────────────────────────────────────
CONFIG = {
    "stockfish_path": "stockfish/stockfish-macos-m1-apple-silicon",
    "model_path": "chess_model.pth",
    "bot_depth": 4,               # Độ sâu minimax của bot
    "matches_per_level": 10,      # Số ván mỗi cấp độ (nên chẵn để xen kẽ màu)
    "sf_move_time": 0.05,         # Thời gian Stockfish suy nghĩ (giây)
    "sf_move_depth": 1,           # Depth Stockfish (dùng kết hợp với time)
    "save_pgn": True,             # Lưu các ván đấu ra file .pgn
    "save_json": True,            # Lưu thống kê ra file .json
    "output_dir": "eval_results", # Thư mục lưu kết quả

    # Các cấp độ sẽ test (Elo hoặc Skill Level)
    # Format: {"label": "...", "elo": int} hoặc {"label": "...", "skill": int}
    "levels": [
        {"label": "Cực yếu",    "skill": 0},
        {"label": "Rất yếu",    "skill": 1},
        {"label": "Yếu",        "skill": 3},
        {"label": "Trung bình", "skill": 5},
        {"label": "Khá",        "skill": 8},
        {"label": "Mạnh",       "elo": 1500},
        {"label": "Rất mạnh",   "elo": 1800},
    ]
}


# ──────────────────────────────────────────────
# DATA CLASSES
# ──────────────────────────────────────────────
@dataclass
class GameResult:
    game_id: int
    level_label: str
    bot_color: str           # "white" hoặc "black"
    result: str              # "BOT_WIN" | "SF_WIN" | "DRAW" | "ERROR"
    termination: str         # "checkmate" | "stalemate" | "repetition" | v.v.
    num_moves: int
    duration_seconds: float
    pgn: str = ""

@dataclass
class LevelStats:
    label: str
    config: dict
    wins: int = 0
    losses: int = 0
    draws: int = 0
    errors: int = 0
    total_moves: int = 0
    total_time: float = 0.0
    games: list = field(default_factory=list)

    @property
    def total(self):
        return self.wins + self.losses + self.draws + self.errors

    @property
    def win_rate(self):
        valid = self.wins + self.losses + self.draws
        return (self.wins / valid * 100) if valid > 0 else 0

    @property
    def score_rate(self):
        """Tính theo điểm: thắng=1, hòa=0.5, thua=0"""
        valid = self.wins + self.losses + self.draws
        return ((self.wins + self.draws * 0.5) / valid * 100) if valid > 0 else 0

    @property
    def avg_moves(self):
        return self.total_moves / max(self.total, 1)


# ──────────────────────────────────────────────
# STOCKFISH CONFIGURATION
# ──────────────────────────────────────────────
def configure_engine(engine: chess.engine.SimpleEngine, level: dict) -> chess.engine.Limit:
    """Cấu hình Stockfish theo level, trả về Limit phù hợp."""
    # Reset về mặc định
    engine.configure({"UCI_LimitStrength": False, "Skill Level": 20})

    if "skill" in level:
        skill = max(0, min(20, level["skill"]))
        engine.configure({"Skill Level": skill})
        limit = chess.engine.Limit(
            depth=CONFIG["sf_move_depth"],
            time=CONFIG["sf_move_time"]
        )
    elif "elo" in level:
        engine.configure({
            "UCI_LimitStrength": True,
            "UCI_Elo": level["elo"],
        })
        limit = chess.engine.Limit(time=0.1)
    else:
        raise ValueError(f"Level không hợp lệ: {level}")

    return limit


def get_level_description(level: dict) -> str:
    if "skill" in level:
        return f"Skill Level {level['skill']}"
    return f"{level['elo']} Elo"


# ──────────────────────────────────────────────
# PLAY SINGLE GAME
# ──────────────────────────────────────────────
def play_game(
    bot,
    engine: chess.engine.SimpleEngine,
    sf_limit: chess.engine.Limit,
    bot_is_white: bool,
    game_id: int,
    level_label: str,
) -> GameResult:
    """Chơi 1 ván, trả về GameResult."""
    board = chess.Board()
    start_time = time.time()
    move_list = []
    bot_color = "white" if bot_is_white else "black"

    try:
        while not board.is_game_over():
            is_bot_turn = (board.turn == chess.WHITE) == bot_is_white

            if is_bot_turn:
                #move = bot.get_best_move(board, depth=CONFIG["bot_depth"])
                move = bot.get_best_move(board)
                if move is None or move not in board.legal_moves:
                    # Bot không tìm được nước đi hợp lệ
                    duration = time.time() - start_time
                    return GameResult(
                        game_id=game_id,
                        level_label=level_label,
                        bot_color=bot_color,
                        result="SF_WIN",
                        termination="bot_illegal_move",
                        num_moves=len(move_list),
                        duration_seconds=round(duration, 2),
                    )
            else:
                sf_result = engine.play(board, sf_limit)
                move = sf_result.move

            move_list.append(board.san(move))
            board.push(move)

    except Exception as e:
        duration = time.time() - start_time
        print(f"\n  ⚠ Lỗi ván {game_id}: {e}")
        return GameResult(
            game_id=game_id,
            level_label=level_label,
            bot_color=bot_color,
            result="ERROR",
            termination=str(e),
            num_moves=len(move_list),
            duration_seconds=round(time.time() - start_time, 2),
        )

    duration = time.time() - start_time

    # Xác định kết quả
    outcome = board.outcome()
    termination = outcome.termination.name.lower() if outcome else "unknown"

    if board.is_checkmate():
        # Người đang đến lượt bị chiếu hết → kẻ thua
        bot_lost = (board.turn == chess.WHITE) == bot_is_white
        result = "SF_WIN" if bot_lost else "BOT_WIN"
    else:
        result = "DRAW"

    # Tạo PGN đơn giản
    pgn = ""
    if CONFIG["save_pgn"]:
        pgn_moves = []
        for i, san in enumerate(move_list):
            if i % 2 == 0:
                pgn_moves.append(f"{i//2 + 1}. {san}")
            else:
                pgn_moves[-1] += f" {san}"
        pgn = " ".join(pgn_moves)

    return GameResult(
        game_id=game_id,
        level_label=level_label,
        bot_color=bot_color,
        result=result,
        termination=termination,
        num_moves=len(move_list),
        duration_seconds=round(duration, 2),
        pgn=pgn,
    )


# ──────────────────────────────────────────────
# EVALUATE ONE LEVEL
# ──────────────────────────────────────────────
def evaluate_level(
    bot,
    stockfish_path: str,
    level: dict,
    num_matches: int,
    game_id_offset: int = 0,
) -> LevelStats:
    """Chạy num_matches ván ở 1 cấp độ."""

    stats = LevelStats(label=level["label"], config=level)
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    try:
        sf_limit = configure_engine(engine, level)
        desc = get_level_description(level)

        print(f"\n{'─'*55}")
        print(f"  📊 {level['label']} ({desc})")
        print(f"{'─'*55}")

        for i in range(num_matches):
            bot_is_white = (i % 2 == 0)  # Xen kẽ màu
            color_symbol = "♔" if bot_is_white else "♚"
            game_id = game_id_offset + i + 1

            print(f"  Ván {i+1:2d}/{num_matches} {color_symbol} ", end="", flush=True)

            result = play_game(
                bot=bot,
                engine=engine,
                sf_limit=sf_limit,
                bot_is_white=bot_is_white,
                game_id=game_id,
                level_label=level["label"],
            )

            # Cập nhật stats
            if result.result == "BOT_WIN":
                stats.wins += 1
                icon = "✅ Thắng"
            elif result.result == "SF_WIN":
                stats.losses += 1
                icon = "❌ Thua"
            elif result.result == "DRAW":
                stats.draws += 1
                icon = "🤝 Hòa"
            else:
                stats.errors += 1
                icon = "⚠ Lỗi"

            stats.total_moves += result.num_moves
            stats.total_time += result.duration_seconds
            stats.games.append(asdict(result))

            print(f"| {icon:10s} | {result.num_moves:3d} nước | {result.duration_seconds:.1f}s")

    finally:
        engine.quit()

    return stats


# ──────────────────────────────────────────────
# PRINT SUMMARY
# ──────────────────────────────────────────────
def print_summary(all_stats: list):
    print("\n")
    print("╔" + "═"*60 + "╗")
    print("║" + "       TỔNG KẾT ĐÁNH GIÁ BOT CHESS".center(60) + "║")
    print("╠" + "═"*60 + "╣")
    print(f"║  {'Cấp độ':<18} {'Thắng':>5} {'Thua':>5} {'Hòa':>5} {'Điểm%':>7} {'Thắng%':>7} ║")
    print("╠" + "═"*60 + "╣")

    for s in all_stats:
        print(f"║  {s.label:<18} {s.wins:>5} {s.losses:>5} {s.draws:>5} "
              f"{s.score_rate:>6.1f}% {s.win_rate:>6.1f}% ║")

    print("╠" + "═"*60 + "╣")

    total_wins = sum(s.wins for s in all_stats)
    total_losses = sum(s.losses for s in all_stats)
    total_draws = sum(s.draws for s in all_stats)
    total_games = sum(s.total for s in all_stats)
    overall_score = sum(s.wins + s.draws * 0.5 for s in all_stats)
    overall_pct = overall_score / max(total_games, 1) * 100

    print(f"║  {'TỔNG CỘNG':<18} {total_wins:>5} {total_losses:>5} {total_draws:>5} "
          f"{overall_pct:>6.1f}%        ║")
    print("╠" + "═"*60 + "╣")

    # Nhận xét
    print("║  📝 NHẬN XÉT:".ljust(61) + "║")

    strong_levels = [s for s in all_stats if s.score_rate >= 50]
    weak_levels   = [s for s in all_stats if s.score_rate < 30]

    if strong_levels:
        labels = ", ".join(s.label for s in strong_levels)
        print(f"║  ✅ Chơi tốt ở  : {labels[:40]:<40} ║")
    if weak_levels:
        labels = ", ".join(s.label for s in weak_levels)
        print(f"║  ❌ Cần cải thiện: {labels[:39]:<39} ║")

    if overall_pct >= 60:
        rating = "🏆 Bot rất mạnh! Thử thách cấp cao hơn."
    elif overall_pct >= 40:
        rating = "📈 Tiến bộ tốt. Tối ưu thêm Minimax/NN."
    elif overall_pct >= 20:
        rating = "🔧 Cần huấn luyện thêm Neural Network."
    else:
        rating = "⚠ Bot còn yếu. Kiểm tra lại model/eval."

    print(f"║  {rating:<58} ║")
    print("╚" + "═"*60 + "╝")


# ──────────────────────────────────────────────
# SAVE RESULTS
# ──────────────────────────────────────────────
def save_results(all_stats: list):
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if CONFIG["save_json"]:
        json_path = os.path.join(CONFIG["output_dir"], f"eval_{timestamp}.json")
        data = {
            "timestamp": timestamp,
            "config": {k: v for k, v in CONFIG.items() if k != "levels"},
            "levels": [
                {
                    "label": s.label,
                    "config": s.config,
                    "wins": s.wins,
                    "losses": s.losses,
                    "draws": s.draws,
                    "errors": s.errors,
                    "win_rate": round(s.win_rate, 2),
                    "score_rate": round(s.score_rate, 2),
                    "avg_moves": round(s.avg_moves, 1),
                    "games": s.games,
                }
                for s in all_stats
            ]
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n  💾 Đã lưu JSON: {json_path}")

    if CONFIG["save_pgn"]:
        pgn_path = os.path.join(CONFIG["output_dir"], f"eval_{timestamp}.pgn")
        with open(pgn_path, "w", encoding="utf-8") as f:
            for s in all_stats:
                for g in s.games:
                    if g.get("pgn"):
                        bot_is_white = g["bot_color"] == "white"
                        white_name = "NNBot" if bot_is_white else "Stockfish"
                        black_name = "Stockfish" if bot_is_white else "NNBot"
                        f.write(f'[Event "Bot Evaluation"]\n')
                        f.write(f'[Level "{s.label}"]\n')
                        f.write(f'[White "{white_name}"]\n')
                        f.write(f'[Black "{black_name}"]\n')
                        f.write(f'[Result "{g["result"]}"]\n\n')
                        f.write(g["pgn"] + "\n\n")
        print(f"  📄 Đã lưu PGN: {pgn_path}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("╔" + "═"*55 + "╗")
    print("║   CHESS BOT EVALUATION - Khởi động...          ║")
    print("╚" + "═"*55 + "╝")
    print(f"  Model   : {CONFIG['model_path']}")
    print(f"  Depth   : {CONFIG['bot_depth']}")
    print(f"  Ván/cấp : {CONFIG['matches_per_level']}")
    print(f"  Cấp độ  : {len(CONFIG['levels'])} levels")
    total_games = len(CONFIG["levels"]) * CONFIG["matches_per_level"]
    print(f"  Tổng ván: ~{total_games} ván đấu")

    # Load bot
    print(f"\n  ⏳ Đang load model...")
    try:
        bot = NNBt(CONFIG["model_path"])
        print(f"  ✅ Load model thành công!")
    except Exception as e:
        print(f"  ❌ Lỗi load model: {e}")
        return

    # Kiểm tra Stockfish
    if not os.path.exists(CONFIG["stockfish_path"]):
        print(f"  ❌ Không tìm thấy Stockfish: {CONFIG['stockfish_path']}")
        return

    all_stats = []
    game_id_offset = 0
    eval_start = time.time()

    for level in CONFIG["levels"]:
        stats = evaluate_level(
            bot=bot,
            stockfish_path=CONFIG["stockfish_path"],
            level=level,
            num_matches=CONFIG["matches_per_level"],
            game_id_offset=game_id_offset,
        )
        all_stats.append(stats)
        game_id_offset += CONFIG["matches_per_level"]

        # Mini summary sau mỗi cấp
        print(f"\n  → Kết quả: {stats.wins}W / {stats.losses}L / {stats.draws}D "
              f"| Score: {stats.score_rate:.1f}% | Avg moves: {stats.avg_moves:.0f}")

    total_time = time.time() - eval_start

    # In tổng kết
    print_summary(all_stats)
    print(f"\n  ⏱ Tổng thời gian: {total_time/60:.1f} phút")

    # Lưu kết quả
    if CONFIG["save_json"] or CONFIG["save_pgn"]:
        save_results(all_stats)


if __name__ == "__main__":
    main()