import chess
import chess.engine
import time
import csv
import math
from datetime import datetime
from bot_VIP import NNBt

# =========================
# CONFIG
# =========================
MODEL_PATH = "chess_model.pth"
STOCKFISH_PATH = "stockfish/stockfish-macos-m1-apple-silicon"

SF_ELO = 2000
NUM_GAMES = 10
BOT_DEPTH = 4
SF_TIME = 0.1

SUMMARY_FILE = "summary_results.csv"


# =========================
# Play one game
# =========================
def play_one_game(bot, engine, bot_is_white):
    board = chess.Board()
    
    while not board.is_game_over():
        is_bot_turn = (board.turn == chess.WHITE) == bot_is_white

        if is_bot_turn:
            move = bot.get_best_move(board)
        else:
            result = engine.play(board, chess.engine.Limit(time=SF_TIME))
            move = result.move

        # Kiểm tra nước đi hợp lệ an toàn hơn
        if move is None or move not in board.legal_moves:
            print(f"Illegal move ({move}) by {'BOT' if is_bot_turn else 'SF'}!")
            return 0 if is_bot_turn else 1

        board.push(move)

    # In kết quả ván đấu ra console để theo dõi
    print(f"  Game ended. Fen: {board.fen()[:30]}... Result: {board.result()}")
    
    result = board.result()
    if result == "1-0":
        return 1 if bot_is_white else 0
    elif result == "0-1":
        return 1 if not bot_is_white else 0
    else:
        return 0.5


# =========================
# MAIN
# =========================
def main():

    bot = NNBt(MODEL_PATH)

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({
        "UCI_LimitStrength": True,
        "UCI_Elo": SF_ELO,
    })

    total_score = 0
    wins = 0
    losses = 0
    draws = 0

    print("===================================")
    print(f"Testing {NUM_GAMES} games vs Stockfish {SF_ELO}")
    print("===================================")

    for i in range(NUM_GAMES):

        bot_is_white = (i % 2 == 0)

        print(f"\nGame {i+1} | Bot is {'White' if bot_is_white else 'Black'}")

        score = play_one_game(bot, engine, bot_is_white)
        total_score += score

        if score == 1:
            wins += 1
            print("Result: BOT WIN")
        elif score == 0:
            losses += 1
            print("Result: BOT LOSS")
        else:
            draws += 1
            print("Result: DRAW")

    engine.quit()

    score_rate = total_score / NUM_GAMES

    # Performance rating
    if 0 < score_rate < 1:
        performance = SF_ELO + 400 * math.log10(score_rate / (1 - score_rate))
        performance = round(performance)
    else:
        performance = "Undefined (0% or 100%)"

    print("\n===================================")
    print("FINAL RESULT")
    print("===================================")
    print(f"Wins  : {wins}")
    print(f"Losses: {losses}")
    print(f"Draws : {draws}")
    print(f"Score : {total_score}/{NUM_GAMES}")
    print(f"Score %: {round(score_rate * 100, 2)}%")
    print(f"Performance Rating: {performance}")

    # =========================
    # Save summary
    # =========================
    file_exists = False
    try:
        with open(SUMMARY_FILE, "r"):
            file_exists = True
    except FileNotFoundError:
        pass

    with open(SUMMARY_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "Date",
                "Stockfish_Elo",
                "Bot_Depth",
                "Games",
                "Wins",
                "Losses",
                "Draws",
                "Score",
                "Score_Percent",
                "Performance_Rating"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            SF_ELO,
            BOT_DEPTH,
            NUM_GAMES,
            wins,
            losses,
            draws,
            total_score,
            round(score_rate * 100, 2),
            performance
        ])

    print(f"\nSummary saved to {SUMMARY_FILE}")


if __name__ == "__main__":
    main()