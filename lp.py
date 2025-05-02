import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt

# Data
A_avail, B_avail, C_avail, D_avail = 1000, 1000, 500, 600
LA, LB, LC, LD = 0.44, 0.35, 0.10, 0.11
HA, HB, HC, HD = 0.12, 0.38, 0.11, 0.39
price_L, price_H = 2.5, 3.5

# Create model
m = gp.Model('Feedsalot')

# Decision variables
x_L = m.addVar(name='x_L', lb=0)
x_H = m.addVar(name='x_H', lb=0)

# Add constraints
cA = m.addConstr(LA*x_L + HA*x_H <= A_avail, name='A')
cB = m.addConstr(LB*x_L + HB*x_H <= B_avail, name='B')
cC = m.addConstr(LC*x_L + HC*x_H <= C_avail, name='C')
cD = m.addConstr(LD*x_L + HD*x_H <= D_avail, name='D')

# Objective: maximize revenue -> minimize negative revenue
m.setObjective(-(price_L*x_L + price_H*x_H), GRB.MINIMIZE)

# Solve
m.optimize()

if m.status == GRB.OPTIMAL:
    xL_val = x_L.X
    xH_val = x_H.X
    opt_obj = -m.ObjVal  # since we minimized negative revenue
    print("Optimal solution:")
    print(f"x_L = {xL_val:.4f} lbs")
    print(f"x_H = {xH_val:.4f} lbs")
    print(f"Optimal Revenue = {opt_obj:.2f}")
    
    # Shadow prices (dual variables)
    print("\nShadow Prices (Dual Values):")
    for c in [cA, cB, cC, cD]:
        print(f"{c.ConstrName}: {c.Pi:.4f}")

# Parametric analysis on the price of H
priceH_values = np.arange(3.0, 4.1, 0.1)
obj_values = []

for pH in priceH_values:
    m_param = gp.Model()
    m_param.setParam('OutputFlag', 0)
    x_Lp = m_param.addVar(lb=0, name='x_L')
    x_Hp = m_param.addVar(lb=0, name='x_H')
    
    # Constraints
    m_param.addConstr(LA*x_Lp + HA*x_Hp <= A_avail)
    m_param.addConstr(LB*x_Lp + HB*x_Hp <= B_avail)
    m_param.addConstr(LC*x_Lp + HC*x_Hp <= C_avail)
    m_param.addConstr(LD*x_Lp + HD*x_Hp <= D_avail)
    
    # Objective for parametric run
    m_param.setObjective(price_L*x_Lp + pH*x_Hp, GRB.MAXIMIZE)
    m_param.optimize()
    
    if m_param.status == GRB.OPTIMAL:
        obj_values.append(m_param.ObjVal)
    else:
        obj_values.append(None)

# Plot results
plt.figure(figsize=(8,5))
plt.plot(priceH_values, obj_values, marker='o')
plt.xlabel('Price of H Feed ($/lb)')
plt.ylabel('Optimal Objective Value ($)')
plt.title('Parametric Analysis: Objective vs H Feed Price')
plt.grid(True)
plt.show()
