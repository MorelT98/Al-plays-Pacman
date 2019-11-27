space = 0
def min_symbols(X, Y, i, j):
    global space
    k = len(X) - 1
    n = len(Y) - 1
    if j > n:
        return 0
    val = float('inf')
    S = set()
    for i_prime in range(i, k + 1):
        if X[i_prime] not in S:
            found = False
            for j_prime in range(j, n + 1):
                if X[i_prime] not in S and Y[j_prime] == X[i_prime]:
                    S.add(X[i_prime])
                    found = True
                    val = min(val, 1 + min_symbols(X, Y, i_prime, j_prime + 1))
            if not found:
                return 0
    space += len(S)
    return val

X = 'PPAPPLE'
Y = 'PENPINEAPPLEAPPLEPEN'
n = min_symbols(X, Y, 0, 0)
print(n)
print(space)
