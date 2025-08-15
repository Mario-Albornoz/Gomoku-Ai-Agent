import numpy as np
from typing import List, Tuple, Optional
from gomoku_game import BOARD_SIZE


class GomokuAgent:
    """
    Alpha–beta Gomoku agent with:
      - Move generation focused around existing stones
      - Move ordering (heuristic pre-sort) for better pruning
      - Terminal detection (win/loss/draw)
      - Numeric, run-based evaluation (streak length + open ends + a few broken-3s)
    """

    # ---- Tunables ----
    DEFAULT_DEPTH = 3
    NEIGHBOR_DISTANCE = 2          # how far around existing stones to generate candidates
    CANDIDATE_LIMIT = 14           # top-K candidates after heuristic ordering (balance speed/strength)

    # Heuristic weights (rough, but effective)
    WIN_5 = 1_000_000
    OPEN_4 = 100_000
    CLOSED_4 = 12_000
    OPEN_3 = 2_000
    BROKEN_3 = 1_200
    CLOSED_3 = 200
    OPEN_2 = 80
    CLOSED_2 = 15
    SINGLE = 2

    def __init__(self, agent_symbol: int, blank_symbol: int, opponent_symbol: int):
        self.name = "AlphaBetaGomokuAgent"
        self.agent_symbol = int(agent_symbol)
        self.blank_symbol = int(blank_symbol)
        self.opponent_symbol = int(opponent_symbol)

    # -------- Public API --------
    def play(self, board: np.ndarray, depth: int = None) -> Tuple[int, int]:
        """
        Returns the (i, j) move chosen by alpha–beta search.
        """
        if depth is None:
            depth = self.DEFAULT_DEPTH

        # If board empty: play center
        if not np.any(board != self.blank_symbol):
            c = BOARD_SIZE // 2
            return (c, c)

        # Top-level search
        value, best_move = self._alphabeta(
            board=board,
            depth=depth,
            alpha=float("-inf"),
            beta=float("inf"),
            maximizing=True
        )

        # Fallback (shouldn't happen, but just in case)
        if best_move is None:
            moves = self.generate_moves(board, distance=self.NEIGHBOR_DISTANCE)
            return moves[0] if moves else (BOARD_SIZE // 2, BOARD_SIZE // 2)
        return best_move

    # -------- Alpha–Beta Search --------
    def _alphabeta(
        self,
        board: np.ndarray,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool
    ) -> Tuple[float, Optional[Tuple[int, int]]]:

        terminal, term_score = self._terminal_evaluation(board)
        if terminal:
            return term_score, None

        if depth == 0:
            return float(self.evaluate_board(board)), None

        # Generate and order moves to improve pruning
        moves = self.generate_moves(board, distance=self.NEIGHBOR_DISTANCE)
        if not moves:
            # No legal moves (should mean draw on a full board)
            return float(self.evaluate_board(board)), None

        ordered = self._order_moves(board, moves, maximizing)
        if self.CANDIDATE_LIMIT and len(ordered) > self.CANDIDATE_LIMIT:
            ordered = ordered[: self.CANDIDATE_LIMIT]

        best_move = None

        if maximizing:
            value = float("-inf")
            for move in ordered:
                i, j = move
                board[i, j] = self.agent_symbol
                child_value, _ = self._alphabeta(board, depth - 1, alpha, beta, False)
                board[i, j] = self.blank_symbol

                if child_value > value:
                    value = child_value
                    best_move = move

                alpha = max(alpha, value)
                if alpha >= beta:
                    break  # prune
            return value, best_move

        else:
            value = float("inf")
            for move in ordered:
                i, j = move
                board[i, j] = self.opponent_symbol
                child_value, _ = self._alphabeta(board, depth - 1, alpha, beta, True)
                board[i, j] = self.blank_symbol

                if child_value < value:
                    value = child_value
                    best_move = move

                beta = min(beta, value)
                if alpha >= beta:
                    break  # prune
            return value, best_move

    # -------- Move Generation & Ordering --------
    def generate_moves(self, board: np.ndarray, distance: int = 1):
        """
        Generate candidate moves near existing stones.
        """
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
            # fall back to any blank if something weird happens
            empties = np.argwhere(board == self.blank_symbol)
            return [tuple(idx) for idx in map(tuple, empties)]

        return list(candidates)

    def _order_moves(self, board: np.ndarray, moves: List[Tuple[int, int]], maximizing: bool) -> List[Tuple[int, int]]:
        """
        Simple move ordering: evaluate the position after each move and sort.
        """
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
            scored.sort(key=lambda x: x[0])  # minimizing prefers lower evals
        return [m for _, m in scored]

    # -------- Terminal / Evaluation --------
    def _terminal_evaluation(self, board: np.ndarray) -> Tuple[bool, float]:
        """
        Detects wins or draw and provides a terminal score if applicable.
        """
        if self._has_five(board, self.agent_symbol):
            return True, float(self.WIN_5)
        if self._has_five(board, self.opponent_symbol):
            return True, float(-self.WIN_5)
        if not np.any(board == self.blank_symbol):  # full board -> draw
            return True, 0.0
        return False, 0.0

    def evaluate_board(self, board: np.ndarray) -> int:
        """
        Symmetric evaluation: player score minus opponent score.
        """
        score = 0
        for line in self._all_lines(board):
            score += self._score_line_numeric(line, self.agent_symbol)
            score -= self._score_line_numeric(line, self.opponent_symbol)
        return score

    # -------- Line Utilities --------
    def _all_lines(self, board: np.ndarray) -> List[np.ndarray]:
        """
        Returns every row, column, and diagonal (both directions) as 1D numpy arrays.
        """
        lines: List[np.ndarray] = []

        # Rows and columns
        for i in range(BOARD_SIZE):
            lines.append(board[i, :])
            lines.append(board[:, i])

        # Main diagonals
        for offset in range(-BOARD_SIZE + 1, BOARD_SIZE):
            diag = np.diagonal(board, offset=offset)
            if diag.size > 0:
                lines.append(diag)

        # Anti-diagonals
        flipped = np.fliplr(board)
        for offset in range(-BOARD_SIZE + 1, BOARD_SIZE):
            adiag = np.diagonal(flipped, offset=offset)
            if adiag.size > 0:
                lines.append(adiag)

        return lines

    def _has_five(self, board: np.ndarray, player: int) -> bool:
        """
        Checks if `player` has any 5-in-a-row on the board.
        """
        p = int(player)
        for line in self._all_lines(board):
            # count consecutive p's
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
        """
        Numeric run-based scoring for a single line for `player`.
        Counts runs of the player's stones and their open ends.
        Also detects a couple of 'broken three' patterns.
        """
        arr = np.array(line, dtype=int)
        n = arr.size
        if n == 0:
            return 0

        score = 0
        i = 0
        B = self.blank_symbol
        P = int(player)

        # Helper: check bounds-safe cell
        def cell(idx: int) -> Optional[int]:
            return arr[idx] if 0 <= idx < n else None

        # Sliding through contiguous runs
        while i < n:
            if arr[i] != P:
                i += 1
                continue

            # Count run length k
            j = i
            while j < n and arr[j] == P:
                j += 1
            k = j - i  # run length
            left = i - 1
            right = j

            left_open = (left >= 0 and arr[left] == B)
            right_open = (right < n and arr[right] == B)
            open_ends = (1 if left_open else 0) + (1 if right_open else 0)

            # Terminal: immediate 5+
            if k >= 5:
                return self.WIN_5

            # Score by run length & open ends
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
            else:  # k == 1
                if open_ends >= 1:
                    score += self.SINGLE

            i = j  # continue after this run

        # Detect "broken threes" like 1 1 _ 1 or 1 _ 1 1 with open ends
        # We look for windows of size 4 that have exactly three P and one B, and are not blocked on both sides.
        for start in range(0, n - 3):
            window = arr[start:start + 4]
            if np.count_nonzero(window == P) == 3 and np.count_nonzero(window == B) == 1:
                # Identify the external neighbors
                left_ext = cell(start - 1)
                right_ext = cell(start + 4)

                # Broken shape must have at least one open end overall (prefer two)
                open_ends = 0
                if left_ext == B:
                    open_ends += 1
                if right_ext == B:
                    open_ends += 1

                if open_ends == 2:
                    score += self.BROKEN_3
                elif open_ends == 1:
                    score += self.CLOSED_3  # weaker than BROKEN_3 but still useful

        return score
