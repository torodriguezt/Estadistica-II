from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _all_regressions(model):
    names = model.model.exog_names[1:]
    X = model.model.exog[:, 1:]
    y = model.model.endog
    n = len(y)
    syy = np.sum((y - y.mean()) ** 2)
    mse_full = model.mse_resid

    rows = []
    for k in range(1, len(names) + 1):
        for idx in combinations(range(len(names)), k):
            Xs = np.column_stack([np.ones(n), X[:, idx]])
            beta = np.linalg.lstsq(Xs, y, rcond=None)[0]
            sse = np.sum((y - Xs @ beta) ** 2)
            p = k + 1
            rows.append({
                "k": k,
                "p": p,
                "R_sq": 1 - sse / syy,
                "adj_R_sq": 1 - (sse / (n - p)) / (syy / (n - 1)),
                "SSE": sse,
                "MSE": sse / (n - p),
                "Cp": sse / mse_full - (n - 2 * p),
                "Variables_in_model": " ".join(names[j] for j in idx)
            })
    table = pd.DataFrame(rows)
    return table.sort_values(["k", "SSE"]).reset_index(drop=True)


def _best_by_size(model):
    table = _all_regressions(model)
    return table.loc[table.groupby("k")["SSE"].idxmin()].reset_index(drop=True)


def _criterion_plot(best, column, label, line=False):
    plt.figure(figsize=(6, 4.5))
    plt.plot(best["p"], best[column], "o-", markersize=10,
             markerfacecolor="white", color="black")
    if line:
        plt.ylim(0, best[column].max() * 1.05)
        plt.axline((0, 0), slope=1, color="red", linestyle="--")
    plt.xticks(best["p"])
    plt.xlabel("p")
    plt.ylabel(label)
    plt.show()


def _print_best(best, column, name, decimals):
    result = best[["k", "p", column, "Variables_in_model"]]
    result = result.rename(columns={column: name,
                                    "Variables_in_model": "Variables.in.model"})
    print("Models are Indexed in rows")
    print(result.to_string(index=False, float_format=f"{{:.{decimals}f}}".format))


def myAllRegTable(model, MSE=False):
    table = _all_regressions(model)
    column = "MSE" if MSE else "SSE"
    table = table[["k", "R_sq", "adj_R_sq", column, "Cp", "Variables_in_model"]]
    table.index = table.index + 1
    return table.round(3)


def myR2_criterion(model):
    best = _best_by_size(model)
    _criterion_plot(best, "R_sq", "R2")
    _print_best(best, "R_sq", "R2", 7)


def myAdj_R2_criterion(model):
    best = _best_by_size(model)
    _criterion_plot(best, "adj_R_sq", "adj_R2")
    _print_best(best, "adj_R_sq", "adjR2", 7)


def myCp_criterion(model):
    best = _best_by_size(model)
    _criterion_plot(best, "Cp", "Cp", line=True)
    _print_best(best, "Cp", "Cp", 6)
