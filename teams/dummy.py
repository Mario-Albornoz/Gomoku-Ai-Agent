import numpy as np
from gomoku_game import BOARD_SIZE

class GomokuAgent:
    def __init__(self, agent_symbol, blank_symbol, opponent_symbol, max_depth=2):
        self.agent_symbol = agent_symbol
        self.blank_symbol = blank_symbol
        self.opponent_symbol = opponent_symbol
        self.name = "dummy"
        self.max_depth = max_depth

    def play(self, board):
        best_score = float('-inf')
        best_move = None
        for move in self.generate_moves(board):
            temp_board = board.copy()
            temp_board[move] = self.agent_symbol
            score = self.alpha_beta(temp_board, self.max_depth - 1, False, float('-inf'), float('inf'))
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    # Alpha-Beta search
    def alpha_beta(self, board, depth, is_agent_turn, alpha, beta):
        if depth == 0:
            return self.evaluate_board(board)

        moves = self.generate_moves(board)
        if is_agent_turn:
            max_eval = float('-inf')
            for move in moves:
                temp_board = board.copy()
                temp_board[move] = self.agent_symbol
                eval = self.alpha_beta(temp_board, depth - 1, False, alpha, beta)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # prune
            return max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                temp_board = board.copy()
                temp_board[move] = self.opponent_symbol
                eval = self.alpha_beta(temp_board, depth - 1, True, alpha, beta)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # prune
            return min_eval

    # Candidate moves near stones
    def generate_moves(self, board, distance=2):
        occupied = np.argwhere(board != self.blank_symbol)
        if len(occupied) == 0:
            center = BOARD_SIZE // 2
            return [(center, center)]
        candidates = set()
        for (i, j) in occupied:
            for dx in range(-distance, distance + 1):
                for dy in range(-distance, distance + 1):
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE:
                        if board[ni, nj] == self.blank_symbol:
                            candidates.add((ni, nj))
        return list(candidates)

    # Simple evaluation using patterns
    def evaluate_board(self, board):
        score = 0
        for line in self.get_all_lines(board):
            score += self.evaluate_line(line, self.agent_symbol)
            score -= self.evaluate_line(line, self.opponent_symbol) // 2
        return score

    # All rows, columns, diagonals
    def get_all_lines(self, board):
        lines = []
        for i in range(BOARD_SIZE):
            lines.append(board[i, :])
            lines.append(board[:, i])
        for i in range(-BOARD_SIZE + 1, BOARD_SIZE):
            lines.append(board.diagonal(i))
            lines.append(np.fliplr(board).diagonal(i))
        return lines

    # Pattern-based scoring
    def evaluate_line(self, line, player):
        line_str = ''.join(str(int(cell)) for cell in line)
        patterns = {
            str(player) * 5: 100000,      # five
            f'0{str(player)*4}0': 10000,  # open four
            f'0{str(player)*3}0': 5000,   # open three
            f'0{str(player)*2}0': 500,    # open two
            str(player): 100
        }
        score = 0
        for pat, val in patterns.items():
            score += line_str.count(pat) * val
        return score






