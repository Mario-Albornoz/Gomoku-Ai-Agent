from typing import List, Set

import numpy as np

from gomoku_game import BOARD_SIZE

class Candidate:
    def __init__(self, i, j, heuristic_value: int = 0):
        self.i = i
        self.j = j
        self.heuristic_value = heuristic_value

    #define to be able to compare for equality later on in the set when generating moves
    def __eq__(self, other):
        return isinstance(other, Candidate) and self.i == other.i and self.j == other.j

    def __hash__(self):
        return hash((self.i, self.j))

    def __repr__(self):
        return f"Candidate(i={self.i}, j={self.j}, h={self.heuristic_value})"


def get_all_lines (board: np.ndarray) -> list:
    lines = []

    for i in range(BOARD_SIZE):
        lines.append(board[i, :])
        lines.append(board[:, i])
    for i in range(-BOARD_SIZE + 1, BOARD_SIZE):
        lines.append(np.fliplr(board).diagonal(i)),
        lines.append(board.diagonal(i))

    return lines


class GomokuAgent:
    def __init__(self, agent_symbol, blank_symbol, opponent_symbol, player_type = "min"):
        self.name : str = __name__
        self.agent_symbol : str = agent_symbol
        self.blank_symbol : str = blank_symbol
        self.opponent_symbol : str = opponent_symbol
        self.player_type : str = "min"
        self.defensive_weight: float = 1.5

    def play(self, board) -> tuple[int, int]:
        candidates = self.generate_moves(board)
        candidate_counter = 0
        best_score = 0
        i : int = 0
        j : int = 0

        for candidate in candidates:
            temp_board = board.copy()
            temp_board[candidate.i , candidate.j] = self.agent_symbol
            score = self.evaluate_board(temp_board)

            print(f'Candidate #{candidate_counter} === {candidate.i}, {candidate.j} : {score}')
            candidate_counter += 1

            if score > best_score:
                best_score = score
                i = candidate.i
                j = candidate.j
        return i, j


    def generate_moves(self, board, distance = 1) -> Set[Candidate]:
        candidates = set()
        occupied = np.argwhere(board != self.blank_symbol)

        if len(occupied) == 0:
            center = BOARD_SIZE // 2
            self.player_type = "max"
            self.defensive_weight = 1.2
            candidates.add(Candidate(center, center))
            return candidates

        #set window search based on the distance, distance = 1 gives a 3 x 3 view around each symbol
        for (i, j) in occupied:
            for dx in range(-distance, distance + 1):
                for dy in range(-distance, distance + 1):
                    ni = i + dx
                    nj = j + dy
                    if 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE:
                        if board[ni][nj] == self.blank_symbol:
                            candidates.add(Candidate(ni, nj))
        return candidates

    def calculate_heuristic(self, candidate: Candidate, board: np.ndarray) -> int:
        heuristic_map = {
            1: 100,
            2: 200,
            3: 500,
            4: 1000,
            5: 10000,
        }
        i, j = candidate.i, candidate.j
        defensive_weight = self.defensive_weight

        player_symbol = self.agent_symbol
        opponent_symbol = self.opponent_symbol

        def check(values, symbol: str) -> int:
            counter = 0
            max_counter = 0
            for value in values:
                if value == symbol:
                    counter += 1
                    max_counter = max(max_counter, counter)
                else:
                    counter = 0
            return max_counter

        lines = get_all_lines(board)

        # offensive_heuristic = sum(
        #     heuristic_map.get(check(line, player_symbol), 0) +
        #     self.evaluate_line(line, player_symbol)
        #     for line in lines
        # )
        #
        # defensive_heuristic = sum(
        #     heuristic_map.get(check(line, opponent_symbol), 0)  +
        #     self.evaluate_line(line, opponent_symbol)
        #     for line in lines
        # )

        offensive_heuristic = 0
        defensive_heuristic = 0
        for line in lines:
            offensive_heuristic += self.evaluate_line(line, player_symbol)
            defensive_heuristic += self.evaluate_line(line, opponent_symbol)

        print("Offensive heuristic: " + str(offensive_heuristic) + "Defensive herusitic: " + str(defensive_heuristic))

        heuristic = int(offensive_heuristic + (defensive_heuristic * defensive_weight))

        return heuristic

    def evaluate_board(self, board: np.ndarray) -> int:
        score = 0
        lines = get_all_lines(board)

        for line in lines:
            score += self.evaluate_line(line, self.agent_symbol)
            score -= self.evaluate_line(line, self.opponent_symbol)

        return score

    def evaluate_line(self, line, player) -> int:
        score = 0
        str_player = str(player)
        str_blank = str(self.blank_symbol)

        str_line = ''.join(str(char) for char in line)

        patterns = {
            str_player * 5: 100000,  # win
            f'{str_blank}{str_player * 4}{str_blank}': 15000,  # open four
            f'{str_blank}{str_player * 4}': 10000,  # one-sided four
            f'{str_player * 4}{str_blank}': 10000,
            f'{str_blank}{str_player * 3}{str_blank}': 800,  # open three
            f'{str_blank}{str_player * 3}': 500,
            f'{str_player * 3}{str_blank}': 500,
            f'{str_blank}{str_player * 2}{str_blank}': 150,  # open two
            f'{str_blank}{str_player * 2}': 100,
            f'{str_player * 2}{str_blank}': 100,
        }

        for pattern, value in patterns.items():
            score += str_line.count(pattern) * value

        return score


if __name__ == '__main__':
    playing_board = np.full((BOARD_SIZE, BOARD_SIZE), ".")
    agent = GomokuAgent(agent_symbol="X", blank_symbol=".", opponent_symbol="O")
    moves = agent.generate_moves(board=playing_board, distance=1)

    for move in moves:
        print(f"Candidate move at ({move.i}, {move.j})")

