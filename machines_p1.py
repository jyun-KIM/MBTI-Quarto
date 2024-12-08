import numpy as np
from itertools import product
import random

class P1():
    def __init__(self, board, available_pieces):
        # 모든 가능한 조각 (4가지 속성을 가진 16개의 조합)
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
        self.board = board  # 4x4 보드 초기화
        self.available_pieces = available_pieces  # 사용 가능한 조각 리스트

    def select_piece(self):
        # 사용 가능한 조각 중 랜덤으로 선택
        return random.choice(self.available_pieces)

    def place_piece(self, selected_piece):
        # 가장 승리 가능성이 높은 위치에 배치
        available_locations = self.get_available_locations()
        winning_move = None
        for loc in available_locations:
            if self.simulate_move(loc, self.pieces.index(selected_piece) + 1):
                winning_move = loc
                break
            
        # 바로 승리할 수 있는 위치가 없으면 점수 기반으로 위치 선택
        if not winning_move:
            winning_move = max(available_locations, key=lambda loc: self.evaluate_position(loc, selected_piece))
        return winning_move

    def simulate_opponent_wins(self, piece):
        # 상대방이 승리할 가능성을 시뮬레이션하여 점수화
        return sum(self.simulate_move(loc, self.pieces.index(piece) + 1) for loc in self.get_available_locations())

    def simulate_move(self, loc, piece_idx):
        # 특정 위치에 조각을 놓아보고, 승리 여부를 시뮬레이션
        self.board[loc[0]][loc[1]] = piece_idx
        win = self.check_win()
        self.board[loc[0]][loc[1]] = 0
        return win

    def check_win(self):
        # 모든 가능한 승리 라인 체크 (가로, 세로, 대각선)
        lines = [
            *self.board,  # 가로
            *self.board.T,  # 세로
            [self.board[i][i] for i in range(4)],  # 좌상-우하 대각선
            [self.board[i][3 - i] for i in range(4)]  # 우상-좌하 대각선
        ]
        return any(self.is_winning_line(line) for line in lines)

    def is_winning_line(self, line):
        # 특정 라인이 승리 조건을 만족하는지 확인
        if 0 in line:
            return False  # 빈 칸이 있으면 승리 불가능
        chars = [self.pieces[idx - 1] for idx in line]
        return any(all(p[i] == chars[0][i] for p in chars) for i in range(4))

    def evaluate_position(self, loc, selected_piece):
        # 위치의 유리함을 평가 (같은 속성이 많은 곳 우선)
        self.board[loc[0]][loc[1]] = self.pieces.index(selected_piece) + 1
        score = sum(self.is_partial_line(line) for line in self.get_lines_containing(loc))
        self.board[loc[0]][loc[1]] = 0
        return score

    def is_partial_line(self, line):
        # 부분적으로 승리 조건을 만족하는지 확인
        if 0 not in line:
            return False
        chars = [self.pieces[idx - 1] for idx in line if idx > 0]
        return len(chars) > 0 and any(all(p[i] == chars[0][i] for p in chars) for i in range(4))

    def get_lines_containing(self, loc):
        # 특정 위치를 포함하는 모든 라인 반환
        r, c = loc
        lines = [
            self.board[r],  # 가로
            self.board[:, c],  # 세로
        ]
        if r == c:
            lines.append([self.board[i][i] for i in range(4)])  # 좌상-우하 대각선
        if r + c == 3:
            lines.append([self.board[i][3 - i] for i in range(4)])  # 우상-좌하 대각선
        return lines

    def get_available_locations(self):
        # 빈 칸의 좌표 반환
        return [(r, c) for r, c in product(range(4), range(4)) if self.board[r][c] == 0]
