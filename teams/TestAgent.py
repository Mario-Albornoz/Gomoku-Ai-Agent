import numpy as np

from gomoku_game import BOARD_SIZE

class Candidate:
    def __init__(self,
                 i:int,
                 j:int,
                 heuristic_value: int
                 ):
            self.i, self.j = i, j
            self.heuristic_value = heuristic_value

class GomokuAgent:
    def __init__(self, agent_symbol, blank_symbol, opponent_symbol):
        self.name = __name__
        self.agent_symbol = agent_symbol
        self.blank_symbol = blank_symbol
        self.opponent_symbol = opponent_symbol
        self.defensive_weight = 0

    def play(self, board) -> tuple[int, int]:
        best_move = tuple()
        candidates = self.generate_move(board)

        highest_heuristic_value = 0
        for candidate in candidates:
            temp_board = board.copy()
            temp_board[candidate.i][candidate.j] = self.agent_symbol
            heuristic_score = self.evaluate_board(temp_board)

            if heuristic_score > highest_heuristic_value:
                highest_heuristic_value = heuristic_score
                best_move = candidate.i , candidate.j

        return best_move

    def generate_move(self, board, distance: int = 1) -> set[Candidate]:
        candidates = set()
        occupied = np.argwhere(board != self.blank_symbol)

        if len(occupied) == 0:
            center = BOARD_SIZE//2
            candidates.add(Candidate(center,center, 0))
            return candidates

        for (i, j) in occupied:
            for dx in range(-distance, distance+1):
                for dy in range(-distance, distance+1):
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE:
                        if board[ni][nj] == self.blank_symbol:
                            candidate = Candidate(ni, nj, 0)
                            candidates.add(candidate)
        return candidates

    def evaluate_board(self, board) -> int:
        lines = self.get_lines(board)
        score = 0

        for line in lines:
            score += self.evaluate_line(line, self.agent_symbol)
            score -= self.evaluate_line(line, self.opponent_symbol)

        return score

    def get_lines(self, board) -> list:
        lines = []

        for i in range(BOARD_SIZE):
            lines.append(board[i, :])
            lines.append(board[:, i])

        for i in range (-BOARD_SIZE + 1, BOARD_SIZE):
            lines.append(board.diagonal(i))
            lines.append(np.fliplr(board).diagonal(i))

        return lines

    def evaluate_line(self, line, player_symbol) -> int:
        heuristic_value = 0
        line_str = ''.join(str(int(cell)) for cell in line)

        patterns = {
            str(player_symbol) * 5: 100000,  # win
            f'0{str(player_symbol) * 4}0': 10000,  # open 4
            f'0{str(player_symbol) * 4}': 9000,
            f'{str(player_symbol) * 4}0': 9000,
            f'0{str(player_symbol) * 3}0': 500,
            f'0{str(player_symbol) * 3}': 400,
            f'{str(player_symbol) * 3}0': 400,
            f'0{str(player_symbol) * 2}0': 100,
            f'0{str(player_symbol) * 2}': 80,
            f'{str(player_symbol) * 2}0': 80,
            str(player_symbol): 10  # single stone
        }

        for pattern, value in patterns.items():
            heuristic_value += line_str.count(pattern) * value

        return heuristic_value