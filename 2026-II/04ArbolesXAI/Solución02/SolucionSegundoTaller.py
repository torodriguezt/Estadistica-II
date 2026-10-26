# Solución del segundo taller de árboles e interpretabilidad.
# Primero se definen las funciones de apoyo; después, la solución de cada punto
# ajusta los modelos con scikit-learn y llama a esas funciones.
# Los datos se cargan con ISLP; la ejecución completa tarda uno o dos minutos.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ISLP import load_data
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (GridSearchCV, PredefinedSplit,
                                     train_test_split, validation_curve)
from sklearn.tree import DecisionTreeRegressor, export_text

SEMILLA = 2026


# =====================================================================
#                          Funciones de apoyo
# =====================================================================
def mse_por_arboles(bosques, X_prueba, y_prueba):
    # MSE de prueba de cada bosque con sus B primeros árboles, B = 1, ..., n_estimators.
    # El bosque con B árboles es el promedio de los B primeros: basta con acumular.
    curvas = {}
    for nombre, bosque in bosques.items():
        pred = pd.DataFrame([arbol.predict(X_prueba.values) for arbol in bosque.estimators_])
        curvas[nombre] = pred.expanding().mean().sub(y_prueba.values).pow(2).mean(axis=1)
    curvas = pd.DataFrame(curvas).set_axis(range(1, len(pred) + 1))
    ax = curvas.plot(xlabel="Número de árboles B", ylabel="MSE de prueba")
    ax.set_ylim(curvas.iloc[9:].min().min() - 1, curvas.iloc[9:].max().max() + 1)
    ax.legend(title="m", ncol=len(bosques), loc="lower center", bbox_to_anchor=(0.5, 1))
    plt.show()
    return curvas


def errores_bosques(bosques, y_entrenamiento, X_prueba, y_prueba):
    # Error out-of-bag (requiere oob_score=True) y MSE de prueba de cada bosque.
    return pd.DataFrame({
        "MSE_OOB": {k: mean_squared_error(y_entrenamiento, b.oob_prediction_)
                    for k, b in bosques.items()},
        "MSE_prueba": {k: mean_squared_error(y_prueba, b.predict(X_prueba))
                       for k, b in bosques.items()},
    })


def tabla_importancia(modelo, X_prueba, y_prueba, semilla=SEMILLA):
    # Importancia por reducción de impureza (entrenamiento, en proporción) y por
    # permutación (aumento promedio del MSE de prueba al permutar cada predictora).
    perm = permutation_importance(modelo, X_prueba, y_prueba,
                                  scoring="neg_mean_squared_error", random_state=semilla)
    tabla = pd.DataFrame({"impureza": modelo.feature_importances_,
                          "permutacion": perm.importances_mean}, index=X_prueba.columns)
    return tabla.sort_values("permutacion", ascending=False)


def curva_lambda(modelo, X, y, lambdas, n_entrenamiento):
    # MSE de entrenamiento y de prueba de un boosting para cada lambda (learning_rate).
    # Las n_entrenamiento primeras filas entrenan y las demás prueban
    # (PredefinedSplit: -1 = siempre entrenamiento, 0 = prueba).
    particion = PredefinedSplit(np.where(np.arange(len(y)) < n_entrenamiento, -1, 0))
    ent, pru = validation_curve(modelo, X, y, param_name="learning_rate",
                                param_range=lambdas, cv=particion,
                                scoring="neg_mean_squared_error")
    curvas = pd.DataFrame({"entrenamiento": -ent[:, 0], "prueba": -pru[:, 0]},
                          index=lambdas)
    ax = curvas.plot(logx=True, marker="o", xlabel=r"$\lambda$", ylabel="MSE")
    ax.axvline(curvas["prueba"].idxmin(), color="gray", linestyle="--")
    plt.show()
    return curvas


def comparar_modelos(modelos, X_ent, y_ent, X_pru, y_pru):
    # Ajusta cada modelo con el entrenamiento y reporta su MSE y su R2 de prueba.
    filas = {}
    for nombre, modelo in modelos.items():
        pred = modelo.fit(X_ent, y_ent).predict(X_pru)
        filas[nombre] = {"MSE": mean_squared_error(y_pru, pred),
                         "R2": r2_score(y_pru, pred)}
    return pd.DataFrame(filas).T


def agregar_ruido(X, semilla=SEMILLA):
    # Agrega ruido_bin ~ Bernoulli(0.5) y ruido_cont ~ N(0, 1), sin relación con Y.
    rng = np.random.default_rng(semilla)
    return X.assign(ruido_bin=rng.binomial(1, 0.5, len(X)),
                    ruido_cont=rng.normal(size=len(X)))


def graficar_dependencia_parcial(modelo, X, variables, ice=False, semilla=SEMILLA):
    # Dependencia parcial: promedio de las predicciones sobre X (method="brute").
    # Con ice=True agrega las curvas ICE de 50 observaciones.
    disp = PartialDependenceDisplay.from_estimator(
        modelo, X, variables, method="brute", kind="both" if ice else "average",
        subsample=50, random_state=semilla)
    plt.show()
    return disp


def graficar_ice_por_grupo(modelo, X, variable, grupo, n=300, semilla=SEMILLA):
    # Curvas ICE centradas de `variable` para n observaciones al azar, coloreadas
    # según `grupo`, con la curva promedio de cada grupo (línea discontinua).
    # Devuelve las curvas promedio centradas, una columna por grupo.
    muestra = X.sample(n, random_state=semilla)
    _, ax = plt.subplots()
    promedios = {}
    for i, valor in enumerate(sorted(muestra[grupo].unique())):
        disp = PartialDependenceDisplay.from_estimator(
            modelo, muestra[muestra[grupo] == valor], [variable], kind="both",
            centered=True, ax=[ax], ice_lines_kw={"color": f"C{i}"},
            pd_line_kw={"color": f"C{i}", "label": f"{grupo} = {valor}"})
        r = disp.pd_results[0]
        promedios[valor] = pd.Series(r["average"][0] - r["average"][0][0],
                                     index=r["grid_values"][0])
    plt.show()
    return pd.DataFrame(promedios)


def evaluar_sustituto(sustituto, modelo, X_prueba, y_prueba):
    # Fidelidad en prueba (R2 de las predicciones del sustituto frente a las del
    # modelo), MSE frente a la respuesta observada y reglas del árbol.
    print(export_text(sustituto, feature_names=list(X_prueba.columns)))
    return {"fidelidad": sustituto.score(X_prueba, modelo.predict(X_prueba)),
            "MSE_frente_a_Y": mean_squared_error(y_prueba, sustituto.predict(X_prueba))}


