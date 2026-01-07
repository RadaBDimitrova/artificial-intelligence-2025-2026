r = -1  # '>'
l = 1   # '<'
m = 0   # '_'

def dfs(state, goal_state, solution, empty):
    solution.append(state[:])
    
    if state == goal_state:
        print('\n'.join(''.join('>' if x == r else '<' if x == l else '_' for x in s) for s in solution))
        return True

    if empty >= 1 and state[empty - 1] == r:
        state[empty - 1], state[empty] = state[empty], state[empty - 1]
        if dfs(state, goal_state, solution, empty - 1):
            return True
        state[empty - 1], state[empty] = state[empty], state[empty - 1]

    if empty >= 2 and state[empty - 2] == r and state[empty - 1] == l:
        state[empty - 2], state[empty] = state[empty], state[empty - 2]
        if dfs(state, goal_state, solution, empty - 2):
            return True
        state[empty - 2], state[empty] = state[empty], state[empty - 2]

    if empty < len(state) - 1 and state[empty + 1] == l:
        state[empty + 1], state[empty] = state[empty], state[empty + 1]
        if dfs(state, goal_state, solution, empty + 1):
            return True
        state[empty + 1], state[empty] = state[empty], state[empty + 1]

    if empty < len(state) - 2 and state[empty + 1] == r and state[empty + 2] == l:
        state[empty + 2], state[empty] = state[empty], state[empty + 2]
        if dfs(state, goal_state, solution, empty + 2):
            return True
        state[empty + 2], state[empty] = state[empty], state[empty + 2]

    solution.pop()
    return False


def main():
    N = int(input())
    start = [r] * N + [m] + [l] * N
    goal_state = [l] * N + [m] + [r] * N
    solution = []
    dfs(start, goal_state, solution, N)

main()
