import gurobipy as gp
from gurobipy import GRB


def n_queens(N):
    model = gp.Model("n_queens")
    
    x = model.addVars(N, N, vtype=GRB.BINARY, name="x")
    y = model.addVars(N, N, vtype=GRB.BINARY, name="y")
    
    model.setObjective(gp.quicksum(y[i, j] for i in range(N) for j in range(N)), GRB.MINIMIZE)
    
    for i in range(N):
        model.addConstr(gp.quicksum(x[i, j] for j in range(N)) == 1, f"row_{i}")
    for j in range(N):
        model.addConstr(gp.quicksum(x[i, j] for i in range(N)) == 1, f"col_{j}")
    
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if k != i:
                    model.addConstr(y[i, j] >= x[k, j], f"row_attack_{i}_{j}_{k}")
            for l in range(N):
                if l != j:
                    model.addConstr(y[i, j] >= x[i, l], f"col_attack_{i}_{j}_{l}")
            for k in range(N):
                for l in range(N):
                    if (k != i or l != j) and abs(i - k) == abs(j - l):
                        model.addConstr(y[i, j] >= x[k, l], f"diag_attack_{i}_{j}_{k}_{l}")

    model.optimize()

    if model.status == GRB.OPTIMAL:
        print(f"Optimal solution found with {model.objVal} attacked positions.")
        print("Queen placements (x[i,j] = 1 if queen at (i, j)):")
        board = [["." for _ in range(N)] for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if x[i, j].x > 0.5:
                    board[i][j] = "Q"
        for row in board:
            print(" ".join(row))

    else:
        print("No optimal solution found.")

if __name__ == '__main__':
    n_queens(8)
