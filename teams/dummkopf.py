import numpy as np
from gomoku_game import BOARD_SIZE

class Node:
    def __init__(self, i:int, j:int, heuristic: int, children:list):
        self.i = i
        self.j = j
        self.heuristic = heuristic
        self.children = children

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

    def generate_moves(self, board, distance=1) -> list:
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

    def get_all_lines(self, board) -> list :
        lines = []

        for i in range(BOARD_SIZE):
            lines.append(board[i, :])  # rows
            lines.append(board[:, i])  # columns

        for i in range(-BOARD_SIZE + 1, BOARD_SIZE):
            lines.append(board.diagonal(i))                   # main diagonals
            lines.append(np.fliplr(board).diagonal(i))        # anti-diagonals

        return lines

    def evaluate_line(self, line, player:int)-> int:
        score = 0
        line_str = ''.join(str(int(cell)) for cell in line)

        patterns = {
            #For these examples assume player symbol = 1
            str(player) * 5: 100000, ## win
            "11111" : 100000,
            f'0{str(player)*4}0': 10000,
            "0" + str(player)*4 + "0": 10000, # open 4
            "011110": 10000,
            f'0{str(player)*3}0': 500,    # open 3
            "0" + str(player)*3 + "0":500,

            f'0{str(player)*2}0': 100,    # open 2
            str(player): 10              # single stone

            #Write found patterns and assign a score to each one
            #run these commands
            # pip install flask
            # flask run
        }

        for pattern, value in patterns.items():
            score += line_str.count(pattern) * value

        return score

    def build_tree(self, current_board, depth: int, is_agent_turn=True) -> Node:
        if depth == 0:
            score = self.evaluate_board(current_board)
            return Node(None, None, score, [])

        moves = self.generate_moves(current_board)
        children = []

        for move in moves:
            temp_board = current_board.copy()
            temp_board[move] = self.agent_symbol if is_agent_turn else self.opponent_symbol
            child_node = self.build_tree(temp_board, depth - 1, not is_agent_turn)
            child_node.i, child_node.j = move
            children.append(child_node)

        score = self.evaluate_board(current_board)
        return Node(None, None, score, children)