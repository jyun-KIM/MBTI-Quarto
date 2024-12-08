import numpy as np
import random
from itertools import product
import math
from copy import deepcopy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.models import load_model
import numpy as np

class MCTSNode:
    def __init__(self, board, available_pieces, piece=None, parent=None, position=None):
        self.board = deepcopy(board)  # 현재 보드 상태
        self.available_pieces = deepcopy(available_pieces)  # 사용 가능한 조각
        self.piece = piece  # 현재 노드에서 사용된 조각
        self.position = position  # 현재 노드에서 배치된 위치
        self.parent = parent  # 부모 노드
        self.children = []  # 자식 노드
        self.wins = 0  # 승리 횟수
        self.visits = 0  # 방문 횟수

    def is_fully_expanded(self):
        return len(self.children) == len(self.available_pieces)

    def best_child(self, exploration_weight=1.4):
        return max(
            self.children,
            key=lambda child: child.wins / (child.visits + 1e-6) +
                              exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
        )
    
    # 노드 확장 
    def expand(self, piece, position=None):

        new_board = deepcopy(self.board)
        print("piece: ", piece)
        print("position: ", position)
        print(self.available_pieces)
        if position and piece in self.available_pieces:
            new_board[position[0]][position[1]] = self.available_pieces.index(piece) + 1
        new_pieces = deepcopy(self.available_pieces)
        print("****")
        if piece in self.available_pieces:
            new_pieces.remove(piece)    
        print("삭제")
        child_node = MCTSNode(new_board, new_pieces, piece, parent=self, position=position)
        self.children.append(child_node)
        # new_pieces.remove(piece)
        return child_node

    # 시뮬레이션 결과를 각 노드에 업데이트 
    def update(self, result):
        self.visits += 1
        # print(type(self.visits), type(self.wins))
        # print("wins: ", self.wins)
        # print("visits: ", self.visits)
        self.wins += result

class P2:

    def __init__(self, board, available_pieces, model_path='quarto_model.h5', env=None):
        # 모델 로드
        self.model = load_model(model_path, compile=False)

        # Quarto 환경 초기화 (env가 전달되지 않은 경우 새로 생성)
        self.env = env if env else QuartoEnvironment()

        # 보드와 사용 가능한 말 초기화
        self.board = board
        self.available_pieces = available_pieces

        # MCTS 초기화
        self.mcts = MCTSWithNN(self.model, self.env)

        # 모든 조각 초기화
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]



    def select_piece(self, simulations=50):
        root = MCTSNode(self.board, self.available_pieces)

        for _ in range(simulations):
            node = root
            while node.is_fully_expanded() and node.children:
                node = node.best_child()

            if not node.is_fully_expanded():
                piece_to_expand = random.choice(node.available_pieces)
                child_node = node.expand(piece_to_expand)
                result = self.mcts.simulate(child_node)  # MCTS의 simulate 호출
                while child_node:
                    child_node.update(result)
                    child_node = child_node.parent

        return root.best_child().piece

    def place_piece(self, selected_piece):
        """
        MCTS를 사용하여 selected_piece를 배치할 가장 좋은 위치를 선택합니다.
        """
        # selected_piece가 유효한지 확인
        print("Before place_piece:")
        print("selected_piece:", selected_piece)
        print("available_pieces:", self.available_pieces)
        print("Selected Piece ID:", id(selected_piece))
        for piece in self.available_pieces:
            print(f"Piece: {piece}, ID: {id(piece)}")

        print("있음")
        if selected_piece not in self.available_pieces:
            print("없음")

        root = MCTSNode(self.board, self.available_pieces)
        simulations = 50

        for _ in range(simulations):
            node = root

            # Selection 단계
            while node.is_fully_expanded() and node.children:
                node = node.best_child()

            # Expansion 단계
            available_locs = [(row, col) for row, col in product(range(4), range(4)) if node.board[row][col] == 0]
            if available_locs:
                position_to_expand = random.choice(available_locs)

                # selected_piece를 사용하여 확장
                child_node = node.expand(selected_piece, position_to_expand)
                # child_node = node.expand(selected_piece, available_locs)

                # Simulation 단계
                result = self.mcts.simulate(child_node)

                # Backpropagation 단계
                while child_node:
                    child_node.update(result)
                    child_node = child_node.parent

        # 최적의 위치 반환
        return root.best_child().position


class QuartoEnvironment:
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.available_pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
        self.done = False
        self.winner = None

    def reset(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.available_pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
        self.done = False
        self.winner = None
        return self.board, self.available_pieces

    def step(self, action):
        piece, position = action
        self.board[position[0]][position[1]] = self.available_pieces.index(piece) + 1
        self.available_pieces.remove(piece)

        if self.check_win(self.board):
            self.done = True
            self.winner = 1  # 현재 플레이어 승리
            return (self.board, self.available_pieces), 1, self.done
        elif not self.available_pieces:
            self.done = True
            self.winner = 0  # 무승부 혹은 실패 
            return (self.board, self.available_pieces), 0, self.done

        return (self.board, self.available_pieces), 0, self.done

    def check_win(self, board):
        for row in range(4):
            if len(set(board[row, :])) == 1 and board[row, 0] != 0:
                return True
        for col in range(4):
            if len(set(board[:, col])) == 1 and board[0, col] != 0:
                return True
        if len(set(board.diagonal())) == 1 and board[0, 0] != 0:
            return True
        if len(set(np.fliplr(board).diagonal())) == 1 and board[0, -1] != 0:
            return True
        return False


class MCTSWithNN:
    def __init__(self, model, env, simulations=50):
        self.model = model
        self.env = env
        self.simulations = simulations


    def simulate(self, node):
        # 현재 보드 상태를 복사
        simulated_board = deepcopy(node.board)
        available_pieces = deepcopy(node.available_pieces)

        # 보드 상태를 모델 입력 형식으로 변환
        board_state = simulated_board.reshape(1, 4, 4, 1)

        # 행동 확률 예측
        action_probs = self.model.predict(board_state)[0]

        # 가능한 위치만 필터링
        available_positions = [(row, col) for row, col in product(range(4), range(4)) if simulated_board[row][col] == 0]
        
        # 빈 위치가 없는 경우 게임 종료 (무승부로 처리)
        if not available_positions:
            print("No available positions. Returning 0 as result (draw).")
            return 0

        # 행동 확률과 가능한 위치를 매핑
        position_probs = np.zeros(len(available_positions))
        for i, (row, col) in enumerate(available_positions):
            position_probs[i] = action_probs[row * 4 + col]

        # 확률이 가장 높은 위치 선택
        best_position_idx = np.argmax(position_probs)
        position = available_positions[best_position_idx]

        # 조각 선택 (랜덤 or 간단한 전략)
        piece = random.choice(available_pieces)

        # 선택한 행동을 보드에 반영
        simulated_board[position[0]][position[1]] = available_pieces.index(piece) + 1
        available_pieces.remove(piece)

        # 승리 여부 평가
        if self.env.check_win(simulated_board):
            return 1  # 승리
        else:
            return 0  # 패배

