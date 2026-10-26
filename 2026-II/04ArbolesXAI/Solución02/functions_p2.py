"""Funciones de apoyo para bosques aleatorios, boosting e interpretación global
de modelos de caja negra (segundo taller de árboles e interpretabilidad).

Uso, con este archivo en la carpeta de trabajo:

    from functions_p2 import (error_por_arboles, error_oob, comparar_modelos,
                              curva_parametro, tabla_importancia,
                              graficar_importancia, importancia_sin_variable,
                              agregar_ruido, graficar_dependencia_parcial,
                              graficar_ice_por_grupo, evaluar_sustituto)

Los modelos se ajustan con scikit-learn (RandomForestRegressor,
GradientBoostingClassifier, ...) y se pasan a estas funciones. Sirven para
regresión y para clasificación: en regresión el error es el MSE y en
clasificación la tasa de error. Los datos deben ser DataFrame y Series de pandas.

    error_por_arboles             error de prueba con los B primeros árboles
    error_oob                     error out-of-bag de un bosque o bagging
    comparar_modelos              ajusta varios modelos y compara su error de prueba
    curva_parametro               error de entrenamiento y de prueba según un parámetro
    tabla_importancia             importancia por impureza y por permutación
    graficar_importancia          gráfico de barras de la importancia
    importancia_sin_variable      importancia por permutación con y sin una predictora
    agregar_ruido                 agrega una predictora binaria y otra continua de ruido
    graficar_dependencia_parcial  dependencia parcial y curvas ICE
    graficar_ice_por_grupo        curvas ICE centradas, coloreadas según un grupo
    evaluar_sustituto             fidelidad, error y reglas de un árbol sustituto
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone, is_classifier
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import PredefinedSplit, validation_curve
from sklearn.tree import export_text

SEMILLA = 2026


# ---------------------------------------------------------------------
#                         Funciones auxiliares
# ---------------------------------------------------------------------
def _nombre_error(modelo):
    return "Tasa de error" if is_classifier(modelo) else "MSE"


def _error(modelo, y, pred):
    # MSE en regresión y tasa de error en clasificación.
    if is_classifier(modelo):
        return 1 - accuracy_score(y, pred)
    return mean_squared_error(y, pred)


def _metrica(modelo):
    # Métrica equivalente de scikit-learn (mayor es mejor).
    return "accuracy" if is_classifier(modelo) else "neg_mean_squared_error"


def _a_error(modelo, puntaje):
    # Convierte el puntaje de _metrica en error.
    return 1 - puntaje if is_classifier(modelo) else -puntaje


def _proba_arbol(arbol, X, n_clases):
    # Probabilidades de un árbol con una columna por clase del ensamble, aunque a
    # su muestra le falte alguna clase (las clases vienen codificadas 0, ..., k - 1).
    prob = np.zeros((len(X), n_clases))
    prob[:, arbol.classes_.astype(int)] = arbol.predict_proba(X)
    return prob


def _error_acumulado(modelo, X, y):
    # Boosting: staged_predict da la predicción tras cada árbol.
    if hasattr(modelo, "staged_predict"):
        return pd.Series([_error(modelo, y, pred) for pred in modelo.staged_predict(X)])
    # Bagging y bosques: el modelo con B árboles promedia los B primeros.
    X = np.asarray(X, dtype=float)
    arboles = modelo.estimators_
    columnas = getattr(modelo, "estimators_features_", [slice(None)] * len(arboles))
    if is_classifier(modelo):
        k = len(modelo.classes_)
        prob = np.array([_proba_arbol(a, X[:, c], k) for a, c in zip(arboles, columnas)])
        prob = prob.cumsum(axis=0) / np.arange(1, len(arboles) + 1)[:, None, None]
        return pd.Series([_error(modelo, y, modelo.classes_[p.argmax(axis=1)])
                          for p in prob])
    pred = pd.DataFrame([a.predict(X[:, c]) for a, c in zip(arboles, columnas)])
    return pred.expanding().mean().sub(np.asarray(y)).pow(2).mean(axis=1)


def _a_flotante(X, variables):
    # La dependencia parcial no admite columnas enteras: se pasan a float.
    nombres = [v for item in variables
               for v in (item if isinstance(item, tuple) else (item,))]
    return X.astype({v: float for v in nombres if pd.api.types.is_integer_dtype(X[v])})


# ---------------------------------------------------------------------
#                    Bosques aleatorios y boosting
# ---------------------------------------------------------------------
def error_por_arboles(modelos, X_prueba, y_prueba):
    """Error de prueba de cada modelo usando solo sus B primeros árboles.

    modelos: un modelo ajustado o un diccionario {nombre: modelo ajustado} de
    bosques aleatorios, bagging o boosting. No hace falta reajustar: en bagging
    y bosques el modelo con B árboles es el promedio de los B primeros, y en
    boosting se usa staged_predict.
    Devuelve un DataFrame (filas: B = 1, 2, ...; columnas: modelos) y grafica
    una curva por modelo.
    """
    if not isinstance(modelos, dict):
        modelos = {"modelo": modelos}
    curvas = pd.DataFrame({nombre: _error_acumulado(m, X_prueba, y_prueba)
                           for nombre, m in modelos.items()})
    curvas.index = curvas.index + 1
    ax = curvas.plot(xlabel="Número de árboles B",
                     ylabel=_nombre_error(next(iter(modelos.values()))) + " de prueba")
    if len(curvas) > 20:  # se recorta el error de los primeros árboles, que es mayor
        resto = curvas.iloc[9:]
        margen = 0.05 * (resto.max().max() - resto.min().min())
        ax.set_ylim(resto.min().min() - margen, resto.max().max() + margen)
    ax.legend(ncol=min(len(modelos), 5), loc="lower center", bbox_to_anchor=(0.5, 1))
    plt.show()
    return curvas


def error_oob(bosque, y_entrenamiento):
    """Error out-of-bag de un bosque aleatorio o bagging ajustado con oob_score=True.

    Cada observación de entrenamiento se predice solo con los árboles que no la
    usaron, así que estima el error de prueba sin separar datos.
    """
    if is_classifier(bosque):
        pred = bosque.classes_[np.argmax(bosque.oob_decision_function_, axis=1)]
    else:
        pred = bosque.oob_prediction_
    return _error(bosque, y_entrenamiento, pred)


def comparar_modelos(modelos, X_ent, y_ent, X_pru, y_pru):
    """Ajusta cada modelo del diccionario {nombre: modelo} con el entrenamiento y
    compara su error de prueba (MSE y R2 en regresión; tasa de error en
    clasificación). Los modelos quedan ajustados.
    """
    filas = {}
    for nombre, modelo in modelos.items():
        pred = modelo.fit(X_ent, y_ent).predict(X_pru)
        filas[nombre] = {_nombre_error(modelo): _error(modelo, y_pru, pred)}
        if not is_classifier(modelo):
            filas[nombre]["R2"] = r2_score(y_pru, pred)
    return pd.DataFrame(filas).T


def curva_parametro(modelo, X_ent, y_ent, X_pru, y_pru, parametro, valores,
                    logx=False):
    """Error de entrenamiento y de prueba del modelo para cada valor de un parámetro.

    parametro: nombre del parámetro de scikit-learn, por ejemplo "learning_rate"
    (lambda en boosting), "n_estimators" (B), "max_leaf_nodes" (d + 1) o
    "max_features" (m en un bosque). El modelo se reajusta para cada valor.
    Devuelve un DataFrame (filas: valores) y grafica las dos curvas; la línea
    vertical marca el valor con menor error de prueba.
    """
    X = pd.concat([X_ent, X_pru])
    y = pd.concat([y_ent, y_pru])
    # -1: siempre en entrenamiento; 0: conjunto de prueba
    particion = PredefinedSplit(np.r_[np.full(len(y_ent), -1), np.zeros(len(y_pru))])
    ent, pru = validation_curve(modelo, X, y, param_name=parametro,
                                param_range=valores, cv=particion,
                                scoring=_metrica(modelo))
    curvas = pd.DataFrame({"entrenamiento": _a_error(modelo, ent[:, 0]),
                           "prueba": _a_error(modelo, pru[:, 0])}, index=valores)
    curvas.index.name = parametro
    ax = curvas.plot(logx=logx, marker="o", ylabel=_nombre_error(modelo))
    ax.axvline(curvas["prueba"].idxmin(), color="gray", linestyle="--")
    plt.show()
    return curvas


# ---------------------------------------------------------------------
#                      Importancia de las variables
# ---------------------------------------------------------------------
def tabla_importancia(modelo, X_prueba, y_prueba, n_repeticiones=5,
                      semilla=SEMILLA):
    """Importancia de cada predictora de un modelo ajustado.

    impureza: proporción de la reducción total de impureza (RSS o Gini) lograda
        en el entrenamiento con cortes en la variable. Solo modelos de árboles;
        favorece a las variables con muchos puntos de corte.
    permutacion: aumento promedio del error de prueba (MSE o tasa de error) al
        permutar la variable, con su desviación estándar entre repeticiones.
    Ordenada de mayor a menor importancia por permutación.
    """
    perm = permutation_importance(modelo, X_prueba, y_prueba,
                                  scoring=_metrica(modelo),
                                  n_repeats=n_repeticiones, random_state=semilla)
    tabla = pd.DataFrame({"permutacion": perm.importances_mean,
                          "desv_permutacion": perm.importances_std},
                         index=X_prueba.columns)
    if hasattr(modelo, "feature_importances_"):
        tabla.insert(0, "impureza", modelo.feature_importances_)
    return tabla.sort_values("permutacion", ascending=False)


def graficar_importancia(tabla, columna="permutacion"):
    """Gráfico de barras horizontales de una columna de tabla_importancia
    ("permutacion", con barras de error, o "impureza")."""
    tabla = tabla.sort_values(columna)
    error = tabla["desv_permutacion"] if columna == "permutacion" else None
    ax = tabla[columna].plot.barh(xerr=error)
    ax.set_xlabel("Aumento del error al permutar" if columna == "permutacion"
                  else "Reducción de impureza (proporción)")
    plt.show()


def importancia_sin_variable(modelo, X_ent, y_ent, X_pru, y_pru, variable,
                             semilla=SEMILLA):
    """Importancia por permutación con todas las predictoras y sin `variable`.

    Reajusta el modelo (sin ajustar) con y sin la variable, imprime el error de
    prueba de ambos y devuelve una tabla con las dos importancias. Sirve para
    ver cómo una predictora muy correlacionada se reparte la importancia con otra.
    """
    completo = clone(modelo).fit(X_ent, y_ent)
    reducido = clone(modelo).fit(X_ent.drop(columns=variable), y_ent)
    for nombre, m, X in [("con", completo, X_pru),
                         ("sin", reducido, X_pru.drop(columns=variable))]:
        print(f"{_nombre_error(m)} de prueba {nombre} {variable}:",
              _error(m, y_pru, m.predict(X)))
    tabla = pd.DataFrame({
        f"con_{variable}": tabla_importancia(completo, X_pru, y_pru,
                                             semilla=semilla)["permutacion"],
        f"sin_{variable}": tabla_importancia(reducido, X_pru.drop(columns=variable),
                                             y_pru, semilla=semilla)["permutacion"]})
    return tabla.sort_values(f"con_{variable}", ascending=False)


def agregar_ruido(X, semilla=SEMILLA):
    """Agrega a X dos predictoras sin relación con la respuesta:
    ruido_bin ~ Bernoulli(0.5) y ruido_cont ~ N(0, 1)."""
    rng = np.random.default_rng(semilla)
    return X.assign(ruido_bin=rng.binomial(1, 0.5, len(X)),
                    ruido_cont=rng.normal(size=len(X)))


# ---------------------------------------------------------------------
#               Dependencia parcial, curvas ICE y sustitutos
# ---------------------------------------------------------------------
def graficar_dependencia_parcial(modelo, X, variables, ice=False, centrado=False,
                                 n_ice=50, semilla=SEMILLA):
    """Dependencia parcial de cada variable de la lista `variables`: promedio de
    las predicciones sobre X al fijar la variable en cada valor de una grilla.

    ice=True agrega las curvas ICE de n_ice observaciones al azar, y
    centrado=True centra todas las curvas en el primer valor de la grilla.
    Una pareja ("x1", "x2") en la lista da la dependencia parcial conjunta (solo
    con ice=False). En clasificación binaria se grafica la probabilidad de la
    segunda clase. Devuelve el PartialDependenceDisplay; los valores calculados
    están en .pd_results.
    """
    disp = PartialDependenceDisplay.from_estimator(
        modelo, _a_flotante(X, variables), variables, method="brute",
        kind="both" if ice else "average", centered=centrado,
        subsample=n_ice, random_state=semilla)
    plt.show()
    return disp


def graficar_ice_por_grupo(modelo, X, variable, grupo, n=300, semilla=SEMILLA):
    """Curvas ICE centradas de `variable` para n observaciones de X al azar,
    coloreadas según la variable categórica `grupo`, con la curva promedio de
    cada grupo (línea discontinua). Si los promedios difieren, hay interacción
    entre `variable` y `grupo`.
    Devuelve las curvas promedio centradas, una columna por grupo.
    """
    muestra = _a_flotante(X, [variable]).sample(n, random_state=semilla)
    # Grilla común a todos los grupos: los valores observados si son pocos; si no,
    # 100 puntos entre los percentiles 5 y 95, como hace scikit-learn.
    x = muestra[variable]
    grilla = (np.unique(x) if x.nunique() < 100
              else np.linspace(*np.quantile(x, [0.05, 0.95]), 100))
    # PartialDependenceDisplay (1.7) busca la grilla por la posición de la
    # columna; se da también por nombre por si eso cambia.
    grilla = {variable: grilla, list(X.columns).index(variable): grilla}
    _, ax = plt.subplots()
    promedios = {}
    for i, valor in enumerate(sorted(muestra[grupo].unique())):
        disp = PartialDependenceDisplay.from_estimator(
            modelo, muestra[muestra[grupo] == valor], [variable], kind="both",
            centered=True, custom_values=grilla, ax=[ax],
            ice_lines_kw={"color": f"C{i}"},
            pd_line_kw={"color": f"C{i}", "label": f"{grupo} = {valor}"})
        r = disp.pd_results[0]
        promedios[valor] = pd.Series(r["average"][0] - r["average"][0][0],
                                     index=r["grid_values"][0])
    plt.show()
    return pd.DataFrame(promedios)


def evaluar_sustituto(sustituto, modelo, X_prueba, y_prueba, reglas=True):
    """Evalúa un árbol sustituto, ajustado a las predicciones del modelo de caja
    negra en el entrenamiento, por ejemplo:
        sustituto = DecisionTreeRegressor(max_depth=3).fit(X_ent, modelo.predict(X_ent))

    fidelidad: R2 (regresión) o proporción de coincidencias (clasificación)
        entre las predicciones del sustituto y las del modelo en prueba.
    error_sustituto y error_modelo: error de prueba frente a la respuesta observada.
    Con reglas=True imprime las reglas del árbol.
    """
    if reglas:
        print(export_text(sustituto, feature_names=list(X_prueba.columns)))
    return {"fidelidad": sustituto.score(X_prueba, modelo.predict(X_prueba)),
            "error_sustituto": _error(modelo, y_prueba, sustituto.predict(X_prueba)),
            "error_modelo": _error(modelo, y_prueba, modelo.predict(X_prueba))}
