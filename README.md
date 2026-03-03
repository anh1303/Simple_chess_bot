# ♟ Simple Chess Bot

Dự án chess bot sử dụng thuật toán Minimax + Alpha-Beta Pruning với nhiều cấp độ engine khác nhau.

## 📁 Cấu trúc dự án

```
├── bot.py          # Bot 1 — Engine cơ bản
├── bot2.py         # Bot 2 — Engine trung bình
├── bot_VIP.py      # Bot 3 — Engine nâng cao (mạnh nhất)
├── test.py         # Đánh giá bot vs Stockfish (nhiều cấp độ)
├── test2.py        # Đánh giá bot vs Stockfish (đơn giản)
├── vs.py           # Chơi trực tiếp với bot qua giao diện Pygame
├── stockfish/      # Thư mục chứa engine Stockfish
└── eval_results/   # Kết quả đánh giá
```

## ⚙️ Cài đặt

```bash
# Tạo virtual environment
python -m venv env
source env/bin/activate   # macOS/Linux

# Cài đặt dependencies
pip install python-chess numpy pygame
```

Tải Stockfish từ [trang chính thức](https://stockfishchess.org/download/) và đặt vào thư mục `stockfish/`.

---

## 🤖 Mô tả 3 Bot

### Bot 1 — `bot.py` (Cơ bản)

| Thành phần | Chi tiết |
|------------|----------|
| **Class** | `NNBt` |
| **Khởi tạo** | `NNBt(model_path=None)` |
| **Tìm nước đi** | `get_best_move(board, depth=3)` |
| **Evaluation** | Material + PST (Pawn, Knight) |
| **Search** | Alpha-Beta + Quiescence Search |
| **Move Ordering** | MVV-LVA + Killer Moves |
| **Độ mạnh** | ⭐ Yếu |

### Bot 2 — `bot2.py` (Trung bình)

| Thành phần | Chi tiết |
|------------|----------|
| **Class** | `NNBt` |
| **Khởi tạo** | `NNBt(model_path=None)` |
| **Tìm nước đi** | `get_best_move(board, depth=3)` |
| **Evaluation** | Material + Bishop Pair + Mobility |
| **Search** | Alpha-Beta + Quiescence + LMR + Transposition Table |
| **Move Ordering** | MVV-LVA + Killer Moves + History Heuristic |
| **Độ mạnh** | ⭐⭐ Trung bình |

### Bot 3 — `bot_VIP.py` (Nâng cao)

| Thành phần | Chi tiết |
|------------|----------|
| **Class** | `NNBt` |
| **Khởi tạo** | `NNBt(book_path=None)` |
| **Tìm nước đi** | `get_best_move(board, depth=3)` |
| **Evaluation** | Material + Full PST (6 loại quân) + Pawn Structure + King Safety + Center Control + Rook Bonuses + Knight Outposts + Space + Endgame Patterns + Pawn Hash Table |
| **Search** | Iterative Deepening + Aspiration Windows + Alpha-Beta + Quiescence + LMR + Null Move Pruning + Futility Pruning + SEE Pruning + Reverse Futility Pruning + ProbCut + Singular Extensions |
| **Move Ordering** | SEE + Killer Moves + Countermove Heuristic + History + Continuation History + Capture History |
| **Extras** | Opening Book (Polyglot), Transposition Table (1M entries), Pawn Hash Table |
| **Độ mạnh** | ⭐⭐⭐ Mạnh |

> **Lưu ý:** Tham số `depth` trong Bot 3 chỉ để đồng bộ interface với Bot 1 và Bot 2, **không được sử dụng**. Bot 3 dùng Iterative Deepening với `max_depth=10` và `time_limit=5.0s`.

---

## 🚀 Cách sử dụng

### Sử dụng chung (cả 3 bot đều giống nhau)

```python
import chess
from bot import NNBt       # Bot 1
# from bot2 import NNBt    # Bot 2
# from bot_VIP import NNBt # Bot 3

# Khởi tạo
bot = NNBt()

# Tìm nước đi tốt nhất
board = chess.Board()
move = bot.get_best_move(board, depth=3)
print(move)  # e.g. e2e4

# Đi nước
board.push(move)
```

### Bot 3 — Tùy chỉnh nâng cao

```python
from bot_VIP import NNBt

# Sử dụng Opening Book
bot = NNBt(book_path="path/to/book.bin")

# Tùy chỉnh độ sâu và thời gian
move = bot.get_best_move(board, max_depth=12, time_limit=10.0)
```

---

## 🧪 Đánh giá bot

### Test đa cấp độ (test.py)

```bash
python test.py
```

Tự động đấu bot với Stockfish ở 7 cấp độ (từ Skill 0 → Elo 1800), xuất kết quả JSON + PGN.

### Test nhanh (test2.py)

```bash
python test2.py
```

Đấu 10 ván với Stockfish Elo 1500, tính Performance Rating.

### Chơi trực tiếp (vs.py)

```bash
python vs.py
```

Giao diện Pygame để chơi trực tiếp với bot (bạn cầm quân Trắng).

---

