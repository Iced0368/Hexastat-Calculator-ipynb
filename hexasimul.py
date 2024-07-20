import numpy as np
import matplotlib.pyplot as plt
import copy
from hexatable import hexa_table, hexa_table_frag, hexa_frag_cost, hex_frag_multiplier, hexa_prob
from hexa_strategy import HexaStrategyModel
from linalg import *

MAX_LEVEL = 10  # 총 단계 수
MAX_ATTEMPTS = 20  # 최대 시도 횟수

np.set_printoptions(precision=4, suppress=True)

def find_strategy(goal_level, high_value_preference, frag_price, reset_cost, is_sunday=False):
    hsm = HexaStrategyModel(goal_level, hexa_frag_cost, hexa_prob[is_sunday], frag_price, reset_cost, MAX_LEVEL, MAX_ATTEMPTS, high_value_preference=high_value_preference)
    
    last_one_indices = []
    for col in range(hsm.should_stop.shape[1]):
        rows_with_ones = np.where(hsm.should_stop[:, col] == 1)[0]
        if rows_with_ones.size > 0:
            last_one_indices.append(rows_with_ones[-1])
        else:
            last_one_indices.append(-1)  # 1이 없는 경우 -1로 표시

    t = '이상' if high_value_preference else '이하'
    print(f'[메인스탯 {goal_level} {t} 목표 시 강화 전략]')
    print()
    prev = -1
    for i in range(10, MAX_ATTEMPTS+1):
        if last_one_indices[i] != prev:
            print(f'{i}회 강화 후 메인스탯 {last_one_indices[i]} 이하면 초기화')
        prev = last_one_indices[i]



def hexa_myulmangjeon(goal_level, high_value_preference, frag_price, reset_cost, is_sunday=False):
    hsm = HexaStrategyModel(goal_level, hexa_frag_cost, hexa_prob[is_sunday], frag_price, reset_cost, MAX_LEVEL, MAX_ATTEMPTS, high_value_preference=high_value_preference)

    num_reinforcements = 1 << 64
    x = {(0, 0): 1} # main-level, attempts

    ######## Apply Strategy ########

    table = hexa_table[is_sunday]
    strategy_table = copy.deepcopy(table)


    for m in range(MAX_LEVEL+1):
        for t in range(MAX_ATTEMPTS+1):
            if hsm.should_stop[m][t]:
                strategy_table.clear_transition((m, t))
                strategy_table.add_transition((m, t), (0, 0), 1.0)


    if high_value_preference:
        for level in range(goal_level, MAX_LEVEL+1):
            strategy_table.clear_transition((level, MAX_ATTEMPTS))
            strategy_table.add_transition((level, MAX_ATTEMPTS), (level, MAX_ATTEMPTS), 1.0)
    else:
        for level in range(0, goal_level+1):
            strategy_table.clear_transition((level, MAX_ATTEMPTS))
            strategy_table.add_transition((level, MAX_ATTEMPTS), (level, MAX_ATTEMPTS), 1.0)

    strategy_table.compile()

    ######## Do Reinforce ########

    result = reinforce(strategy_table, x, num_reinforcements)

    result_vector = np.zeros(MAX_LEVEL+1)
    for i in range(MAX_LEVEL+1):
        result_vector[i] = result[(i,)]


    ######## Log & Visualize ########

    fig, ax = plt.subplots(figsize=(10, 3))
    bars = ax.bar(range(len(result_vector)), result_vector)

    # 각 막대 위에 값 표시
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{round(100*yval, 2)}%', ha='center', va='bottom')

    # y축 범위 설정
    ax.set_ylim(0, 1)
    plt.show()


    def cost_branch(state, hsm, only_frag=False):
        i, j = state
        if hsm.should_stop[i][j]:
            return reset_cost if not only_frag else 0
        elif j == 20:
            return 0
        else:
            return hexa_frag_cost[i] * hex_frag_multiplier * frag_price if not only_frag else hexa_frag_cost[i] * hex_frag_multiplier


    strategy_cost = strategy_table.create_vector({
        (i, j) : cost_branch((i, j), hsm) for i, j in strategy_table.states
    })

    strategy_frag_cost = strategy_table.create_vector({
        (i, j) : cost_branch((i, j), hsm, True) for i, j in strategy_table.states
    })

    xvector = strategy_table.create_vector(x)
    migs = matrix_semi_inf_geometric_series(strategy_table.get_matrix())

    expected_cost = xvector @ migs @ strategy_cost.T
    expected_frag_cost = xvector @ migs @ strategy_frag_cost.T

    for i, p in enumerate(result_vector):
        if p > 0:
            print(f'메인 {i:3d}레벨 확률: {p*100:2.2f}%')

    print()

    print(f"평균 총 메소 소모량\t: {round(expected_cost, 2)} 억 메소")
    print(f"평균 조각 소모량\t\t: {round(expected_frag_cost, 2)} 개")


def hexa_try(goal_level, num_frags, high_value_preference, frag_price, reset_cost, is_sunday=False):
    hsm = HexaStrategyModel(goal_level, hexa_frag_cost, hexa_prob[is_sunday], frag_price, reset_cost, MAX_LEVEL, MAX_ATTEMPTS, high_value_preference=high_value_preference)

    x = {(0, 0, 0): 1} # main-level, attempts, spent-frags

    ######## Apply Strategy ########

    table = hexa_table_frag[is_sunday]
    strategy_table = copy.deepcopy(table)

    init_states = []
    for fsstate, prob in strategy_table.transitions.items():
        from_state, to_state = fsstate
        m, t, f = to_state
        if hsm.should_stop[m][t] and f == 0:
            init_states.append((from_state, to_state, prob))

    for from_state, to_state, prob in init_states:
        del strategy_table.transitions[(from_state, to_state)]
        prev_prob = strategy_table.transitions.get((from_state, (0, 0, 0)), 0)
        strategy_table.add_transition(from_state, (0, 0, 0), prev_prob + prob)


    if high_value_preference:
        for level in range(goal_level, MAX_LEVEL+1):
            strategy_table.clear_transition((level, MAX_ATTEMPTS, 0))
            strategy_table.add_transition((level, MAX_ATTEMPTS, 0), (level, MAX_ATTEMPTS, 0), 1.0)   
    else:
        for level in range(0, goal_level+1):
            strategy_table.clear_transition((level, MAX_ATTEMPTS, 0))
            strategy_table.add_transition((level, MAX_ATTEMPTS, 0), (level, MAX_ATTEMPTS, 0), 1.0)

    strategy_table.compile()

    result_vector = np.zeros(MAX_LEVEL+1)

    ######## Do Reinforce ########
    if high_value_preference:
        result = reinforce(strategy_table, x, num_frags // hex_frag_multiplier)

        for i in range(MAX_LEVEL+1):
            result_vector[i] = result[(i,)]

    else:
        result = reinforce(strategy_table, x, num_frags // hex_frag_multiplier, False)
        result = strategy_table.compress_state_dict(strategy_table.vector_to_state_dict(result), [0, 1])
        for i in range(MAX_LEVEL+1):
            result_vector[i] = result.get((i,MAX_ATTEMPTS), 0)

    ######## Log & Visualize ########

    fig, ax = plt.subplots(figsize=(10, 3))
    bars = ax.bar(range(len(result_vector)), result_vector)

    # 각 막대 위에 값 표시
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{round(100*yval, 2)}%', ha='center', va='bottom')

    # y축 범위 설정
    ax.set_ylim(0, 1)
    plt.show()

    for i, p in enumerate(result_vector):
        if p > 0:
            print(f'메인 {i:3d}레벨 확률: {p*100:2.2f}%')

    print()

    if high_value_preference:
        print('성공률:', str(round(100*np.sum(result_vector[goal_level:]), 2)) + '%')
    else:
        print('성공률:', str(round(100*np.sum(result_vector[:goal_level+1]), 2)) + '%')