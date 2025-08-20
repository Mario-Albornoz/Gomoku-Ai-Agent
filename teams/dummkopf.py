import time
from copy import deepcopy
from venv import create

import numpy as np
from typing import List, Tuple, Optional
from gomoku_game import BOARD_SIZE

class Node:
    def __init__(self, i:int, j:int, heuristic: int, children:list):
        self.i = i
        self.j = j
        self.heuristic = heuristic
        self.children = children

class GomokuAgent:
    """
    Alpha–beta Gomoku agent with:
      - Move generation focused around existing stones
      - Move ordering (heuristic pre-sort) for better pruning
      - Transposition table with Zobrist hashing
      - Terminal detection (win/loss/draw)
      - Numeric, run-based evaluation (streak length + open ends + a few broken-3s)
    """

    #TODO: Add TSS instead of just pure alpha beta prunning
    #TODO: Add better broken pattern detection and double broken threes
    #TODO: Fix intersection pattern detection and play

    # ---- Tunables ----
    DEFAULT_DEPTH = 3
    NEIGHBOR_DISTANCE = 2
    CANDIDATE_LIMIT = 14

    # Heuristic weights
    WIN_5 = 1_000_000
    OPEN_4 = 100_000
    CLOSED_4 = 12_000
    OPEN_3 = 2_200
    BROKEN_3 = 1_200
    CLOSED_3 = 200
    OPEN_2 = 80
    CLOSED_2 = 15
    SINGLE = 2

    def __init__(self, agent_symbol: int, blank_symbol: int, opponent_symbol: int):
        self.name = "Dummkopf"
        self.agent_symbol = int(agent_symbol)
        self.blank_symbol = int(blank_symbol)
        self.opponent_symbol = int(opponent_symbol)

        # ---- Zobrist hashing setup ----
        rng = np.random.default_rng(12345)
        self.zobrist_table = rng.integers(
            low=1, high=2**63,
            size=(BOARD_SIZE, BOARD_SIZE, 3),  # 3 possible states per cell
            dtype=np.int64
        )
        self.transposition_table = {}  # cache: hash -> {depth, value, best_move}

    # -------- Public API --------
    def play(self, board: np.ndarray, depth: int = None) -> Tuple[int, int]:
        start_time = time.time_ns()
        if depth is None:
            depth = self.DEFAULT_DEPTH

        if not np.any(board != self.blank_symbol):
            c = BOARD_SIZE // 2
            return (c, c)


        attacker_symbol = self.agent_symbol
        defender_symbol = self.opponent_symbol

        # tss_result = self.tss_search(board, attacker_symbol, defender_symbol)
        # if tss_result:
        #     best_move = tss_result[0]
        #     print("TSS found forced win sequence:", tss_result)
        #     return best_move


        self.current_hash = self.compute_hash(board)
        value, best_move = self._alphabeta(board, depth, float("-inf"), float("inf"), True)

        if best_move is None:
            moves = self.generate_moves(board, distance=self.NEIGHBOR_DISTANCE)
            return moves[0] if moves else (BOARD_SIZE // 2, BOARD_SIZE // 2)

        end_time = time.time_ns()
        total_time = (end_time - start_time) / 1_000_000
        print(f"Play took: {total_time} ms")

        return best_move

    # -------- Hashing Utilities --------
    def piece_index(self, val: int) -> int:
        if val == self.blank_symbol:
            return 0
        elif val == self.agent_symbol:
            return 1
        else:  # opponent
            return 2

    def compute_hash(self, board: np.ndarray) -> int:
        h = np.int64(0)
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                piece = self.piece_index(board[i, j])
                h ^= self.zobrist_table[i, j, piece]
        return int(h)

    def update_hash(self, h: int, i: int, j: int, piece: int) -> int:
        """
        Incrementally update hash:
        - remove blank
        - add piece
        """
        old_piece = self.piece_index(self.blank_symbol)
        new_piece = self.piece_index(piece)
        h ^= int(self.zobrist_table[i, j, old_piece])  # remove old
        h ^= int(self.zobrist_table[i, j, new_piece])  # add new
        return h

    def undo_hash(self, h: int, i: int, j: int, piece: int) -> int:
        """
        Undo move (piece -> blank).
        """
        old_piece = self.piece_index(piece)
        new_piece = self.piece_index(self.blank_symbol)
        h ^= int(self.zobrist_table[i, j, old_piece])
        h ^= int(self.zobrist_table[i, j, new_piece])
        return h

    # -------- Alpha–Beta Search --------
    def _alphabeta(self, board: np.ndarray, depth: int, alpha: float, beta: float, maximizing: bool) -> Tuple[float, Optional[Tuple[int, int]]]:
        global child_hash
        if self.current_hash in self.transposition_table:
            entry = self.transposition_table[self.current_hash]
            if entry["depth"] >= depth:
                print("Hashed Entry Returned")
                return entry["value"], entry["best_move"]

        terminal, term_score = self._terminal_evaluation(board)
        if terminal:
            return term_score, None
        if depth == 0:
            return float(self.evaluate_board(board)), None

        moves = self.generate_moves(board, distance=self.NEIGHBOR_DISTANCE)
        if not moves:
            return float(self.evaluate_board(board)), None

        ordered = self._order_moves(board, moves, maximizing)
        if self.CANDIDATE_LIMIT and len(ordered) > self.CANDIDATE_LIMIT:
            ordered = ordered[: self.CANDIDATE_LIMIT]

        best_move = None
        if maximizing:
            value = float("-inf")
            for i, j in ordered:
                board[i, j] = self.agent_symbol
                prev_hash = self.current_hash
                self.current_hash = self.update_hash(prev_hash, i, j, self.agent_symbol)

                child_value, _ = self._alphabeta(board, depth - 1, alpha, beta, False)

                # capture child position hash BEFORE undo
                child_hash = self.current_hash

                board[i, j] = self.blank_symbol
                self.current_hash = prev_hash  # restore hash

                if child_value > value:
                    value = child_value
                    best_move = (i, j)

                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:
            value = float("inf")
            for i, j in ordered:
                board[i, j] = self.opponent_symbol
                prev_hash = self.current_hash
                self.current_hash = self.update_hash(prev_hash, i, j, self.opponent_symbol)

                child_value, _ = self._alphabeta(board, depth - 1, alpha, beta, True)

                child_hash = self.current_hash

                board[i, j] = self.blank_symbol
                self.current_hash = prev_hash

                if child_value < value:
                    value = child_value
                    best_move = (i, j)

                beta = min(beta, value)
                if alpha >= beta:
                    break

        self.transposition_table[child_hash] = {
            "depth": depth,
            "value": value,
            "best_move": best_move,
        }

        return value, best_move

    def tss_search(self, board, attacker_symbol, defender_symbol, depth=0, max_depth=10):
        if depth >= max_depth:
            return False

        threats = self.identify_threats(board, attacker_symbol)

        for t in threats:
            if t["type"] == "double":
                return [t["move"]]

        for t in threats:
            if t["type"]== "single":
                move = t["move"]

                new_board = deepcopy(board)
                new_board[move[0], move[1]] = attacker_symbol

                defense_moves = self.get_defensive_moves(new_board, defender_symbol, move, attacker_symbol)

                if not defense_moves:

                    return [move]

                all_fail = True
                for d in defense_moves:
                    defended_board = deepcopy(new_board)
                    defended_board[d[0], d[1]] = defender_symbol

                    result = self.tss_search(defended_board, attacker_symbol, defender_symbol, depth + 1, max_depth)

                    if result:
                        return [move, d] + result
                    else:
                        all_fail = False


                if all_fail:
                    return [move]

            return []



    # -------- Move Generation & Ordering --------
    def generate_moves(self, board: np.ndarray, distance: int = 1):
        candidates = set()
        occupied = np.argwhere(board != self.blank_symbol)

        if len(occupied) == 0:
            center = BOARD_SIZE // 2
            return [(center, center)]

        for (i, j) in occupied:
            for dx in range(-distance, distance + 1):
                for dy in range(-distance, distance + 1):
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE:
                        if board[ni, nj] == self.blank_symbol:
                            candidates.add((ni, nj))

        if not candidates:
            empties = np.argwhere(board == self.blank_symbol)
            return [tuple(idx) for idx in map(tuple, empties)]

        return list(candidates)

    def get_defensive_moves(self, board, defender_symbol, last_attacker_move, attacker_symbol, blank_symbol=0):
        defense_moves = []
        lines = self._lines_through_cell(board, last_attacker_move)

        for line, index_map in lines:
            if self.creates_open_x(line, attacker_symbol, 4):
                defense_moves.extend(self._open_end_blocks_in_line(line, attacker_symbol, 4, index_map))

            elif self.creates_open_x(line, attacker_symbol, 3):
                defense_moves.extend(self._open_end_blocks_in_line(line, attacker_symbol, 3, index_map))

        defense_moves = list(set(defense_moves))
        return defense_moves

    def identify_threats(self, board: np.ndarray, attacker_symbol: int):
        threats = []
        moves = self.generate_moves(board)
        for (i, j) in moves:
            temp_board = deepcopy(board)
            temp_board[i , j] = attacker_symbol
            lines = self._all_lines(temp_board)

            threat_count = 0

            for line in lines:
                if self.creates_open_x(line, attacker_symbol, 4):
                    threat_count += 1
                if self.creates_open_x(line,attacker_symbol,3):
                    threat_count += 1

            if threat_count == 1:
                threats.append({"move": (i,j), "type": "single" ,"threat_count":threat_count})
            if threat_count >= 2:
                threats.append({"move": (i,j), "type": "double" ,"threat_count": threat_count})
        return threats

    #x represents the number of symbols in a row in an open pattern
    def creates_open_x(self, line, attacker: int, x: int):
        window_size = x + 2
        for start in range(len(line) - window_size + 1):
            window = line[start:start + window_size]

            if window[0] == self.blank_symbol and window[-1] == self.blank_symbol:
                if np.sum(window[1:-1] == attacker) == x and np.all(window[1:-1] == attacker):
                    return True
        return False

    def _lines_through_cell(self, board: np.ndarray, cell: Tuple[int, int]):
        i, j = cell
        n = board.shape[0]
        out = []

        row_map = [(i, y) for y in range(n)]
        out.append((board[i, :].copy(), row_map))

        col_map = [(x, j) for x in range(n)]
        out.append((board[:, j].copy(), col_map))


        d_min = -min(i, j)
        d_max = min(n - 1 - i, n - 1 - j)
        diag_coords = [(i + d, j + d) for d in range(d_min, d_max + 1)]
        diag_line = np.array([board[x, y] for x, y in diag_coords], dtype=board.dtype)
        out.append((diag_line, diag_coords))


        d_min = -min(i, n - 1 - j)
        d_max = min(n - 1 - i, j)
        anti_coords = [(i + d, j - d) for d in range(d_min, d_max + 1)]
        anti_line = np.array([board[x, y] for x, y in anti_coords], dtype=board.dtype)
        out.append((anti_line, anti_coords))

        return out

    def _open_end_blocks_in_line(
            self,
            line: np.ndarray,
            attacker_symbol: int,
            x: int,
            index_map: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:

        blocks = []
        B = self.blank_symbol
        w = x + 2

        L = len(line)
        for start in range(0, L - w + 1):
            window = line[start:start + w]
            if window[0] == B and window[-1] == B and np.all(window[1:-1] == attacker_symbol):
                blocks.append(index_map[start])
                blocks.append(index_map[start + w - 1])
        return blocks

    def get_defense_moves(
            self,
            board: np.ndarray,
            last_attacker_move: Tuple[int, int],
            attacker_symbol: int
    ) -> List[Tuple[int, int]]:
        defenses = set()

        for line, idx_map in self._lines_through_cell(board, last_attacker_move):
            for x in (4, 3):
                blocks = self._open_end_blocks_in_line(line, attacker_symbol, x, idx_map)
                for b in blocks:
                    defenses.add(b)

        return list(defenses)

    def _order_moves(self, board: np.ndarray, moves: List[Tuple[int, int]], maximizing: bool) -> List[Tuple[int, int]]:
        scored: List[Tuple[float, Tuple[int, int]]] = []
        if maximizing:
            for (i, j) in moves:
                board[i, j] = self.agent_symbol
                s = self.evaluate_board(board)
                board[i, j] = self.blank_symbol
                scored.append((s, (i, j)))
            scored.sort(key=lambda x: x[0], reverse=True)
        else:
            for (i, j) in moves:
                board[i, j] = self.opponent_symbol
                s = self.evaluate_board(board)
                board[i, j] = self.blank_symbol
                scored.append((s, (i, j)))
            scored.sort(key=lambda x: x[0])
        return [m for _, m in scored]

    # -------- Terminal / Evaluation --------
    def _terminal_evaluation(self, board: np.ndarray) -> Tuple[bool, float]:
        if self._has_five(board, self.agent_symbol):
            return True, float(self.WIN_5)
        if self._has_five(board, self.opponent_symbol):
            return True, float(-self.WIN_5)
        if not np.any(board == self.blank_symbol):
            return True, 0.0
        return False, 0.0

    def evaluate_board(self, board: np.ndarray) -> int:
        score = 0
        for line in self._all_lines(board):
            score += self._score_line_numeric(line, self.agent_symbol)
            score -= self._score_line_numeric(line, self.opponent_symbol)
        return score

    # -------- Line Utilities --------
    def _all_lines(self, board: np.ndarray) -> List[np.ndarray]:
        lines: List[np.ndarray] = []
        for i in range(BOARD_SIZE):
            lines.append(board[i, :])
            lines.append(board[:, i])
        for offset in range(-BOARD_SIZE + 1, BOARD_SIZE):
            diag = np.diagonal(board, offset=offset)
            if diag.size > 0:
                lines.append(diag)
        flipped = np.fliplr(board)
        for offset in range(-BOARD_SIZE + 1, BOARD_SIZE):
            adiag = np.diagonal(flipped, offset=offset)
            if adiag.size > 0:
                lines.append(adiag)
        return lines

    def _has_five(self, board: np.ndarray, player: int) -> bool:
        p = int(player)
        for line in self._all_lines(board):
            count = 0
            for v in line:
                if v == p:
                    count += 1
                    if count >= 5:
                        return True
                else:
                    count = 0
        return False

    def _score_line_numeric(self, line: np.ndarray, player: int) -> int:
        arr = np.array(line, dtype=int)
        n = arr.size
        if n == 0:
            return 0

        score = 0
        i = 0
        B = self.blank_symbol
        P = int(player)

        def cell(idx: int) -> Optional[int]:
            return arr[idx] if 0 <= idx < n else None

        while i < n:
            if arr[i] != P:
                i += 1
                continue

            j = i
            while j < n and arr[j] == P:
                j += 1
            k = j - i
            left = i - 1
            right = j

            left_open = (left >= 0 and arr[left] == B)
            right_open = (right < n and arr[right] == B)
            open_ends = (1 if left_open else 0) + (1 if right_open else 0)

            if k >= 5:
                return self.WIN_5

            if k == 4:
                if open_ends == 2:
                    score += self.OPEN_4
                elif open_ends == 1:
                    score += self.CLOSED_4
            elif k == 3:
                if open_ends == 2:
                    score += self.OPEN_3
                elif open_ends == 1:
                    score += self.CLOSED_3
            elif k == 2:
                if open_ends == 2:
                    score += self.OPEN_2
                elif open_ends == 1:
                    score += self.CLOSED_2
            else:
                if open_ends >= 1:
                    score += self.SINGLE

            i = j

        for start in range(0, n - 3):
            window = arr[start:start + 4]
            if np.count_nonzero(window == P) == 3 and np.count_nonzero(window == B) == 1:
                left_ext = cell(start - 1)
                right_ext = cell(start + 4)
                open_ends = 0
                if left_ext == B:
                    open_ends += 1
                if right_ext == B:
                    open_ends += 1
                if open_ends == 2:
                    score += self.BROKEN_3
                elif open_ends == 1:
                    score += self.CLOSED_3

        return score
