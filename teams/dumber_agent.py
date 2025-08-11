import numpy as np
import math
from gomoku_game import BOARD_SIZE


class GomokuAgent:
    def __init__(self, agent_symbol, blank_symbol, opponent_symbol):
        self.name = __name__
        self.agent_symbol = agent_symbol
        self.blank_symbol = blank_symbol
        self.opponent_symbol = opponent_symbol

    def play(self, board):
        moves = self.generate_moves(board)
        best_score = float('-inf')
        best_move = None

        for move in moves:
            temp_board = board.copy()
            temp_board[move] = self.agent_symbol
            score = self.evaluate_board(temp_board)

            print(f'Move: {move}, score: {score}')

            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def generate_moves(self, board, distance=1):
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
        return list(candidates)

    def evaluate_board(self, board):
        score = 0
        lines = self.get_all_lines(board)

        for line in lines:
            score += self.evaluate_line(line, self.agent_symbol)
            score -= self.evaluate_line(line, self.opponent_symbol)
        return score

    def get_all_lines(self, board):
        lines = []

        for i in range(BOARD_SIZE):
            lines.append(board[i, :])  # rows
            lines.append(board[:, i])  # columns

        for i in range(-BOARD_SIZE + 1, BOARD_SIZE):
            lines.append(board.diagonal(i))                   # main diagonals
            lines.append(np.fliplr(board).diagonal(i))        # anti-diagonals

        return lines

    def evaluate_line(self, line, player):
        score = 0
        line_str = ''.join(str(int(cell)) for cell in line)

        patterns = {
            str(player) * 5: 100000,     # win
            f'0{str(player)*4}0': 10000,  # open 4
            f'0{str(player)*3}0': 500,    # open 3
            f'0{str(player)*2}0': 100,    # open 2
            str(player): 10              # single stone
        }

        for pattern, value in patterns.items():
            score += line_str.count(pattern) * value

        return score
