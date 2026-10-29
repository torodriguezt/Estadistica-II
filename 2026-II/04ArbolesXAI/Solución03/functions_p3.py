"""Funciones de apoyo para bosques aleatorios, boosting e interpretación global y
local de modelos de caja negra (talleres segundo y tercero de árboles e
interpretabilidad). Reúne todas las funciones de functions_p2.py y agrega las
de los métodos locales.

Uso, con este archivo en la carpeta de trabajo:

    from functions_p3 import (error_por_arboles, error_oob, comparar_modelos,
                              curva_parametro, tabla_importancia,
                              graficar_importancia, importancia_sin_variable,
                              agregar_ruido, graficar_dependencia_parcial,
                              graficar_ice_por_grupo, evaluar_sustituto,
                              valores_coalicion, shapley_por_ordenes,
                              explicar_shap, graficar_cascada,
                              graficar_resumen_shap, graficar_dependencia_shap,
                              importancia_shap, lime_tabular, estabilidad_lime,
                              ice_individual, contrafactual_ice)

Los modelos se ajustan con scikit-learn (RandomForestRegressor,
GradientBoostingClassifier, ...) y se pasan a estas funciones. Sirven para
regresión y para clasificación: en regresión el error es el MSE y en
clasificación la tasa de error. En los métodos locales la predicción es la
respuesta estimada en regresión y la probabilidad de la segunda clase en
clasificación binaria. Los datos deben ser DataFrame y Series de pandas.

    Bosques aleatorios, boosting e interpretación global (segundo taller)
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

    Interpretación local (tercer taller)
    observacion_cercana_umbral   selecciona el cliente más próximo al umbral
    valores_coalicion             función de valor v(C) de x0 para todas las coaliciones
    shapley_por_ordenes           contribuciones en los p! órdenes; su promedio da los phi_j
    verificar_eficiencia          tabla de base, aportes y predicción; verifica la igualdad
    comparar_shap                tabla del cálculo manual y las explicaciones de SHAP
    explicar_shap                 valores SHAP exactos con la librería shap
    graficar_cascada              gráfico de cascada de una observación
    graficar_resumen_shap         gráfico de resumen (beeswarm) de varias observaciones
    graficar_dependencia_shap     phi_j de una predictora frente a su valor
    importancia_shap              media de |phi_j| de cada predictora
    lime_tabular                  explicación LIME de una observación
    estabilidad_lime              LIME con varias semillas y anchos de núcleo
    resumir_lime                  resumen por ancho: coeficientes, fidelidad y tamaño efectivo
    comparar_importancias        tabla de SHAP, permutación e impureza
    ice_individual                curva ICE de una observación
    contrafactual_ice             cruces de una curva ICE con un umbral
"""
from itertools import combinations, permutations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone, is_classifier
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LinearRegression
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


def _prediccion(modelo, X):
    # Probabilidad de la segunda clase en clasificación binaria; predicción en regresión.
    if is_classifier(modelo):
        return modelo.predict_proba(X)[:, 1]
    return modelo.predict(X)


def _fila(x0):
    # La observación a explicar como Series, venga como DataFrame de una fila o no.
    return x0.iloc[0] if isinstance(x0, pd.DataFrame) else x0


def _coalicion(variables):
    # Nombre de una coalición: "{}", "{x1}", "{x1, x2}", ...
    return "{" + ", ".join(variables) + "}"


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
                      semilla=SEMILLA, metrica=None):
    """Importancia de cada predictora de un modelo ajustado.

    impureza: proporción de la reducción total de impureza (RSS o Gini) lograda
        en el entrenamiento con cortes en la variable. Solo modelos de árboles;
        favorece a las variables con muchos puntos de corte.
    permutacion: aumento promedio de la pérdida de prueba al permutar la
        variable, con su desviación estándar entre repeticiones.
    metrica: puntaje de scikit-learn que define la pérdida. Por defecto, la tasa
        de error ("accuracy") en clasificación y el MSE en regresión; con
        "neg_brier_score" o "neg_log_loss" se mide en la escala de la probabilidad.
    Ordenada de mayor a menor importancia por permutación.
    """
    perm = permutation_importance(modelo, X_prueba, y_prueba,
                                  scoring=metrica or _metrica(modelo),
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


# ---------------------------------------------------------------------
#                 Valores de Shapley y SHAP (métodos locales)
# ---------------------------------------------------------------------
def observacion_cercana_umbral(modelo, X, umbral=0.5):
    """Devuelve una fila de X: la predicción más cercana al umbral indicado.

    En clasificación binaria usa la probabilidad de la segunda clase. Conserva
    el índice y las columnas de X, de modo que el resultado puede pasarse
    directamente a valores_coalicion, explicar_shap y estabilidad_lime.
    En caso de empate selecciona la primera fila. No usa la respuesta observada.
    """
    posicion = np.argmin(np.abs(_prediccion(modelo, X) - umbral))
    return X.iloc[[posicion]]


def valores_coalicion(modelo, x0, X_ref):
    """Función de valor v(C) de la observación x0 para todas las coaliciones C.

    v(C) es el promedio de las predicciones sobre X_ref cuando las predictoras
    de C toman los valores de x0 y las demás conservan los de cada fila de
    X_ref (el conjunto de referencia). v({}) es la predicción media sobre X_ref
    y v(todas) = f(x0). Son 2^p coaliciones, así que solo sirve con pocas
    predictoras. x0 es un DataFrame de una fila (o una Series).
    Devuelve una Series indexada por el nombre de la coalición: "{}", "{x1}",
    "{x1, x2}", ..., con las variables en el orden de las columnas de X_ref.
    """
    x0 = _fila(x0)
    variables = list(X_ref.columns)
    v = {}
    for k in range(len(variables) + 1):
        for C in combinations(variables, k):
            Z = X_ref.assign(**{c: x0[c] for c in C})
            v[_coalicion(C)] = _prediccion(modelo, Z).mean()
    return pd.Series(v)


def shapley_por_ordenes(v, variables):
    """Contribución de cada predictora en cada uno de los p! órdenes de revelación.

    v: la función de valor de valores_coalicion; variables: las columnas en el
    mismo orden. En un orden, la contribución de X_j es v(C U {j}) - v(C), con C
    las predictoras reveladas antes que ella. Los valores de Shapley son el
    promedio de cada columna, shapley_por_ordenes(v, variables).mean(), y por la
    eficiencia cada fila suma v(todas) - v({}) = f(x0) - phi_0.
    Devuelve un DataFrame (filas: órdenes; columnas: predictoras).
    """
    variables = list(variables)
    filas = {}
    for orden in permutations(variables):
        fila, reveladas = {}, []
        for j in orden:
            antes = _coalicion([c for c in variables if c in reveladas])
            reveladas.append(j)
            fila[j] = v[_coalicion([c for c in variables if c in reveladas])] - v[antes]
        filas[", ".join(orden)] = fila
    return pd.DataFrame(filas).T[variables].rename_axis("orden")


def verificar_eficiencia(v, ordenes):
    """Comprueba la eficiencia de los órdenes de revelación y la resume.

    Recibe los resultados de valores_coalicion y shapley_por_ordenes. Devuelve
    una Series con la base, la suma de aportes, la predicción y la diferencia
    de reconstrucción (cero salvo redondeo). También verifica cada orden.
    """
    base = v[_coalicion([])]
    prediccion = v[_coalicion(ordenes.columns)]
    np.testing.assert_allclose(ordenes.sum(axis=1), prediccion - base,
                               rtol=0, atol=1e-10)
    suma = ordenes.mean().sum()
    return pd.Series({"base": base, "suma_aportes": suma,
                      "prediccion": prediccion,
                      "diferencia": base + suma - prediccion})


def comparar_shap(v, ordenes, **explicaciones):
    """Compara el cálculo manual con una o más explicaciones de un cliente.

    Ejemplo: comparar_shap(v, ordenes, shap=shap_x0, shap_100=shap_x0_100).
    Devuelve una tabla con una columna por método y una fila por predictora,
    más la base phi_0. Verifica que cada columna reconstruya la predicción;
    las atribuciones pueden diferir si las referencias son distintas.
    """
    eficiencia = verificar_eficiencia(v, ordenes)
    tabla = pd.DataFrame({"manual": ordenes.mean()})
    tabla.loc["phi_0"] = eficiencia["base"]
    for nombre, explicacion in explicaciones.items():
        if len(explicacion.values) != 1:
            raise ValueError("Cada explicación debe corresponder a una sola observación.")
        columna = pd.Series(explicacion.values[0], index=explicacion.feature_names)
        columna.loc["phi_0"] = explicacion.base_values[0]
        tabla[nombre] = columna
    np.testing.assert_allclose(tabla.sum(axis=0), eficiencia["prediccion"],
                               rtol=0, atol=1e-10)
    return tabla


def explicar_shap(modelo, X, X_ref, n_ref=None, semilla=SEMILLA):
    """Valores SHAP exactos de cada fila de X con la librería shap.

    Usa la misma función de valor de valores_coalicion (shap.maskers.Independent):
    las predictoras ausentes toman los valores de cada fila de la referencia,
    X_ref completa o, con n_ref, una muestra al azar de n_ref filas (el costo
    crece con el tamaño de la referencia). shap.explainers.Exact evalúa las 2^p
    coaliciones, así que solo es viable con pocas predictoras.
    Devuelve un shap.Explanation: .values (phi_j, una columna por predictora),
    .base_values (phi_0 = v({})) y .data (los valores de X). Verifica que la
    base y los aportes reconstruyan la predicción de cada fila.
    """
    if n_ref is not None:
        X_ref = X_ref.sample(n_ref, random_state=semilla)

    def f(Z):  # shap pasa arreglos de numpy
        return _prediccion(modelo, pd.DataFrame(Z, columns=X_ref.columns))

    masker = shap.maskers.Independent(X_ref, max_samples=len(X_ref))
    explicacion = shap.explainers.Exact(f, masker)(X, silent=True)
    np.testing.assert_allclose(explicacion.base_values + explicacion.values.sum(axis=1),
                               _prediccion(modelo, X), rtol=0, atol=1e-10)
    return explicacion


def graficar_cascada(explicacion, i=0):
    """Gráfico de cascada de la fila i de una explicación de explicar_shap: parte
    de phi_0 = E[f(X)] y suma los phi_j hasta llegar a f(x)."""
    plt.figure()
    shap.plots.waterfall(explicacion[i], show=False)
    plt.show()


def graficar_resumen_shap(explicacion, semilla=SEMILLA):
    """Gráfico de resumen (beeswarm): un punto por observación y predictora, en la
    posición de su phi_j y coloreado según el valor de la predictora. Las
    predictoras se ordenan por la media de |phi_j|."""
    np.random.seed(semilla)
    _, ax = plt.subplots()
    shap.plots.beeswarm(explicacion, ax=ax, plot_size=None,
                        color_bar_label="Valor de la predictora", show=False)
    ax.set_xlabel("Valor SHAP")
    plt.show()


def graficar_dependencia_shap(explicacion, variable, color=None):
    """Gráfico de dependencia: phi_j de `variable` frente a su valor, un punto por
    observación, coloreado según la predictora `color`. La separación vertical de colores puede sugerir interacciones,
    pero también depende de las otras predictoras y de la referencia."""
    _, ax = plt.subplots()
    shap.plots.scatter(explicacion[:, variable],
                       color=explicacion[:, color] if color else "#1E88E5",
                       ax=ax, show=False)
    ax.set_ylabel(f"Valor SHAP de {variable}")
    plt.show()


def importancia_shap(explicacion):
    """Media de |phi_j| sobre las filas explicadas, de mayor a menor: cuánto mueve
    cada predictora la predicción, en promedio y en sus mismas unidades. No usa
    la respuesta observada."""
    return pd.Series(np.abs(explicacion.values).mean(axis=0),
                     index=explicacion.feature_names).sort_values(ascending=False)


def comparar_importancias(modelo, X_prueba, y_prueba, explicacion,
                          n_repeticiones=5, semilla=SEMILLA):
    """Tabla conjunta de media de |SHAP|, permutación e impureza.

    SHAP se promedia sobre los clientes de explicacion; la permutación usa
    X_prueba e y_prueba. Reporta el aumento del error y su desviación entre
    repeticiones. En clasificación binaria agrega el puntaje de Brier;
    en regresión el error es el MSE. Incluye impureza si el modelo la ofrece.
    """
    perm = tabla_importancia(modelo, X_prueba, y_prueba,
                             n_repeticiones=n_repeticiones, semilla=semilla)
    tabla = pd.DataFrame({"media_abs_shap": importancia_shap(explicacion),
                          "permutacion_error": perm["permutacion"],
                          "desv_error": perm["desv_permutacion"]})
    if is_classifier(modelo) and len(modelo.classes_) == 2:
        brier = tabla_importancia(modelo, X_prueba, y_prueba,
                                  n_repeticiones=n_repeticiones, semilla=semilla,
                                  metrica="neg_brier_score")
        tabla["permutacion_brier"] = brier["permutacion"]
        tabla["desv_brier"] = brier["desv_permutacion"]
    if "impureza" in perm:
        tabla["impureza"] = perm["impureza"]
    return tabla.sort_values("media_abs_shap", ascending=False)


# ---------------------------------------------------------------------
#                LIME, curvas ICE individuales y contrafactuales
# ---------------------------------------------------------------------
def lime_tabular(modelo, x0, X_ref, ancho, M=3000, semilla=SEMILLA):
    """Explicación LIME de x0 con un modelo lineal ponderado.

    1. Genera M perturbaciones z_i: cada predictora toma, de forma independiente,
       un valor al azar de su columna en X_ref.
    2. Estandariza alrededor de x0: z~_ij = (z_ij - x0_j) / s_j, con s_j la
       desviación estándar de X_j en X_ref.
    3. Pondera con el núcleo pi(z_i) = exp(-D_i^2 / ancho^2), con D_i la distancia
       euclídea entre z_i y x0 en esa escala (aquí se comparan tres anchos).
    4. Ajusta f(z_i) = b_0 + sum_j b_j z~_ij por mínimos cuadrados ponderados.
    Devuelve una Series con el intercepto (aproxima f(x0)), los coeficientes
    (cambio de la predicción por cada desviación estándar de X_j cerca de x0),
    el R2 ponderado (fidelidad local) y el tamaño efectivo de la muestra,
    (sum pi)^2 / sum pi^2: cuántas perturbaciones pesan realmente en el ajuste.
    """
    x0 = _fila(x0)[X_ref.columns]
    rng = np.random.default_rng(semilla)
    Z = pd.DataFrame({c: rng.choice(X_ref[c].to_numpy(), M) for c in X_ref.columns})
    Z_est = (Z - x0) / X_ref.std()
    pesos = np.exp(-(Z_est ** 2).sum(axis=1) / ancho ** 2)
    pred = _prediccion(modelo, Z)
    lineal = LinearRegression().fit(Z_est, pred, sample_weight=pesos)
    return pd.Series({"intercepto": lineal.intercept_,
                      **dict(zip(X_ref.columns, lineal.coef_)),
                      "R2": lineal.score(Z_est, pred, sample_weight=pesos),
                      "n_efectivo": pesos.sum() ** 2 / (pesos ** 2).sum()})


def estabilidad_lime(modelo, x0, X_ref, anchos, variable, n_semillas=10, M=3000,
                     semilla=SEMILLA):
    """Repite lime_tabular con n_semillas semillas (semilla, semilla + 1, ...) para
    cada ancho de núcleo de la lista `anchos`.

    Devuelve un DataFrame con una fila por (ancho, semilla) y grafica el
    coeficiente de `variable` de cada repetición (puntos) y su promedio (línea)
    por ancho. Una nube angosta indica una explicación estable frente al azar
    del muestreo.
    """
    tabla = pd.DataFrame({(a, semilla + s): lime_tabular(modelo, x0, X_ref, a, M, semilla + s)
                          for a in anchos for s in range(n_semillas)}).T
    tabla = tabla.rename_axis(["ancho", "semilla"])
    _, ax = plt.subplots()
    for i, (ancho, grupo) in enumerate(tabla.groupby(level="ancho")):
        ax.plot(i + np.linspace(-0.12, 0.12, len(grupo)), grupo[variable], "o",
                color=f"C{i}", alpha=0.7)
        ax.hlines(grupo[variable].mean(), i - 0.25, i + 0.25, color=f"C{i}")
    ax.set_xticks(range(len(anchos)), [f"{a:.2f}" for a in sorted(anchos)])
    ax.set_xlabel(r"Ancho del núcleo $\sigma$")
    ax.set_ylabel(f"Coeficiente de {variable}")
    plt.show()
    return tabla


def resumir_lime(resultados, variable):
    """Resume la tabla de estabilidad_lime por ancho de núcleo.

    Devuelve media, desviación estándar, mínimo y máximo del coeficiente de
    variable entre semillas, y promedios del intercepto, R2 y tamaño efectivo.
    """
    return resultados.groupby(level="ancho").agg(
        media=(variable, "mean"), desv=(variable, "std"), minimo=(variable, "min"),
        maximo=(variable, "max"), intercepto=("intercepto", "mean"),
        R2=("R2", "mean"), n_efectivo=("n_efectivo", "mean"))


def contrafactual_ice(curva, valor_actual, umbral=0.5, objetivo="menor"):
    """Candidato de una curva ICE que cumple un objetivo respecto al umbral.

    La Series debe tener una grilla única, finita e incluir valor_actual.
    objetivo='menor' busca predicción < umbral; 'mayor_igual' busca >= umbral.
    Se devuelve el candidato evaluado más cercano y su predicción, o None si
    no hay uno. Si el valor actual ya cumple, el cambio necesario es cero.
    intervalos_cruce delimita los cambios de lado observados; cruces y
    cruce_cercano son sus puntos medios orientativos, no candidatos verificados.
    El mínimo solo se refiere a la grilla, también si hay múltiples cruces.
    """
    # La grilla debe incluir el valor actual. Los cruces son intervalos, no saldos
    # evaluados: el candidato es el valor observado más cercano que cumple el
    # objetivo (< umbral o >= umbral). Si ya se cumple, el cambio necesario es cero.
    curva = curva.sort_index()
    if objetivo not in {"menor", "mayor_igual"}:
        raise ValueError("objetivo debe ser 'menor' o 'mayor_igual'.")
    if not curva.index.is_unique or valor_actual not in curva.index:
        raise ValueError("La grilla debe ser única e incluir valor_actual.")
    if not np.isfinite(curva.to_numpy()).all() or not np.isfinite(curva.index).all():
        raise ValueError("La curva debe contener valores finitos.")
    arriba = (curva >= umbral).to_numpy()
    x = curva.index.to_numpy(dtype=float)
    i = np.flatnonzero(arriba[1:] != arriba[:-1])
    cruces = (x[i] + x[i + 1]) / 2
    cumple = ~arriba if objetivo == "menor" else arriba
    candidatos = x[cumple]
    candidato = (float(candidatos[np.argmin(np.abs(candidatos - valor_actual))])
                 if len(candidatos) else None)
    return {"cruces": cruces.tolist(),
            "intervalos_cruce": list(zip(x[i].tolist(), x[i + 1].tolist())),
            "cruce_cercano": (float(cruces[np.argmin(np.abs(cruces - valor_actual))])
                              if len(cruces) else None),
            "contrafactual": candidato,
            "cambio": candidato - float(valor_actual) if candidato is not None else None,
            "probabilidad": float(curva.loc[candidato]) if candidato is not None else None,
            "ya_cumple": bool(cumple[curva.index.get_loc(valor_actual)])}


def ice_individual(modelo, x0, variable, grilla, umbral=None):
    """Evalúa y grafica la predicción al cambiar una sola variable de x0.

    Mantiene fijas las demás columnas, ordena la grilla e incluye el valor
    exacto de x0. Devuelve una Series con las predicciones evaluadas. Si se
    indica umbral, dibuja la frontera aproximada más cercana, sin confundirla
    con el candidato que devuelve contrafactual_ice para un objetivo concreto.
    """
    # Curva ICE de x0: predicción al mover `variable` por la grilla con las demás
    # predictoras fijas en x0. Marca x0 y, si se da, el umbral y el cruce más cercano.
    x0 = _fila(x0)
    grilla = np.unique(np.append(grilla, x0[variable]))
    Z = pd.DataFrame([x0] * len(grilla)).assign(**{variable: grilla})
    curva = pd.Series(_prediccion(modelo, Z), index=pd.Index(grilla, name=variable))
    _, ax = plt.subplots()
    curva.plot(ax=ax, ylabel="Probabilidad estimada" if is_classifier(modelo) else "Predicción")
    ax.plot(x0[variable], _prediccion(modelo, x0.to_frame().T)[0], "o", label="$x_0$")
    if umbral is not None:
        ax.axhline(umbral, color="gray", linestyle="--")
        cruce = contrafactual_ice(curva, x0[variable], umbral)["cruce_cercano"]
        if cruce is not None:
            ax.axvline(cruce, color="gray", linestyle=":")
    ax.legend()
    plt.show()
    return curva
