"""
optimiser.py - ILP Squad Optimiser
Author: Dhruv Mehta
"""
import pulp
import pandas as pd

def optimise_squad(players, budget=100.0, pred_col="predicted_points"):
    """
    Select optimal 15-player FPL squad using Integer Linear Programming.
    Constraints: budget, positional limits, max 3 per club.
    """
    n = len(players)
    indices = list(range(n))
    prob = pulp.LpProblem("FPL_Squad", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in indices]

    prob += pulp.lpSum(players.iloc[i][pred_col] * x[i] for i in indices)
    prob += pulp.lpSum(players.iloc[i]["price"] * x[i] for i in indices) <= budget
    prob += pulp.lpSum(x[i] for i in indices) == 15

    for pos, count in [("GKP",2),("DEF",5),("MID",5),("FWD",3)]:
        prob += pulp.lpSum(x[i] for i in indices
                           if players.iloc[i]["position"] == pos) == count

    for club in players["team"].unique():
        prob += pulp.lpSum(x[i] for i in indices
                           if players.iloc[i]["team"] == club) <= 3

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    selected = [i for i in indices if pulp.value(x[i]) == 1]
    return players.iloc[selected].copy()

def squad_summary(squad, pred_col="predicted_points"):
    print(f"Total Price: £{squad['price'].sum():.1f}m")
    print(f"Total Predicted: {squad[pred_col].sum():.1f} pts")
    if "actual_points" in squad.columns:
        print(f"Total Actual: {squad['actual_points'].sum():.0f} pts")
    for pos in ["GKP","DEF","MID","FWD"]:
        sub = squad[squad["position"]==pos][["name","team","price",pred_col]]
        print(f"\n--- {pos} ---")
        print(sub.to_string(index=False))
