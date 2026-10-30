"""Funciones de apoyo para árboles de regresión y de clasificación (primer
taller de árboles e interpretabilidad).

Uso, con este archivo en la carpeta de trabajo:

    from functions_p1 import (rss, gini, evaluar_cortes, mejor_corte,
                              graficar_cortes, graficar_respuesta,
                              graficar_arbol, tabla_hojas, predecir_con_ruta,
                              graficar_regiones, errores_regresion,
                              curva_profundidad, matriz_confusion,
                              metricas_clasificacion)

Los árboles se ajustan con scikit-learn (DecisionTreeRegressor y
DecisionTreeClassifier) y se pasan a estas funciones. Las funciones de búsqueda
de cortes repiten a mano lo que hace el árbol en cada nodo. Los datos deben ser
DataFrame y Series de pandas.

    Búsqueda de cortes
    rss                     RSS de un grupo al predecirlo con su media
    gini                    índice de Gini de un grupo
    evaluar_cortes          RSS o Gini ponderado de cada corte admisible de una predictora
    mejor_corte             mejor corte de cada predictora frente a no dividir
    graficar_cortes         criterio de cada corte admisible frente al punto de corte

    Lectura del árbol ajustado
    graficar_respuesta      respuesta frente a cada predictora
    graficar_arbol          dibujo del árbol con plot_tree
    tabla_hojas             regla, tamaño y predicción de cada hoja
    predecir_con_ruta       hoja, regla y predicción de observaciones nuevas
    graficar_regiones       partición del plano de dos predictoras

    Evaluación
    errores_regresion       RSS, MSE y RMSE de entrenamiento y de prueba
    curva_profundidad       error de entrenamiento, validación cruzada y prueba por profundidad
    matriz_confusion        matriz de confusión con la clase positiva al final
    metricas_clasificacion  TCC, tasa de error, sensibilidad y especificidad
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.base import clone, is_classifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, cross_validate
from sklearn.tree import plot_tree

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


def _reglas(modelo):
    # Condiciones que llevan desde la raíz hasta cada nodo. scikit-learn envía
    # a la izquierda los valores <= corte y numera a cada hijo después de su padre.
    arbol = modelo.tree_
    nombres = getattr(modelo, "feature_names_in_",
                      [f"X{j}" for j in range(modelo.n_features_in_)])
    reglas = {0: []}
    for nodo in range(arbol.node_count):
        if arbol.children_left[nodo] >= 0:
            v, c = nombres[arbol.feature[nodo]], arbol.threshold[nodo]
            reglas[arbol.children_left[nodo]] = reglas[nodo] + [f"{v} <= {c:g}"]
            reglas[arbol.children_right[nodo]] = reglas[nodo] + [f"{v} > {c:g}"]
    return reglas


# ---------------------------------------------------------------------
#                          Búsqueda de cortes
# ---------------------------------------------------------------------
def rss(y):
    """Suma de cuadrados de los residuales de un grupo al predecirlo con su media."""
    y = np.asarray(y, dtype=float)
    return float(((y - y.mean()) ** 2).sum())


def gini(y):
    """Índice de Gini de un grupo: 1 - sum_k p_k^2, con p_k la proporción de la clase k."""
    p = pd.Series(np.asarray(y)).value_counts(normalize=True)
    return float(1 - (p ** 2).sum())


def evaluar_cortes(x, y, min_hoja=10, criterio="rss"):
    """Evalúa todos los cortes admisibles de la predictora numérica x.

    Los candidatos son los puntos medios entre valores consecutivos distintos de
    x: cualquier otro punto entre ellos deja las mismas observaciones a cada
    lado. Se admiten los cortes que dejan al menos min_hoja observaciones en
    cada grupo; el grupo izquierdo es x < corte.
    criterio="rss" (regresión): RSS de cada grupo y RSS total.
    criterio="gini" (clasificación): Gini de cada grupo y Gini ponderado,
        (n_izq * G_izq + n_der * G_der) / n.
    Devuelve un DataFrame con una fila por corte admisible; la última columna es
    el criterio que se minimiza.
    """
    medida = {"rss": rss, "gini": gini}[criterio]
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    valores = np.unique(x)
    filas = []
    for corte in (valores[:-1] + valores[1:]) / 2:
        izq = x < corte
        n_izq, n_der = int(izq.sum()), int((~izq).sum())
        if min(n_izq, n_der) < min_hoja:
            continue
        m_izq, m_der = medida(y[izq]), medida(y[~izq])
        total = (m_izq + m_der if criterio == "rss"
                 else (n_izq * m_izq + n_der * m_der) / len(x))
        filas.append((corte, n_izq, n_der, m_izq, m_der, total))
    columnas = ["corte", "n_izq", "n_der"] + (
        ["RSS_izq", "RSS_der", "RSS_total"] if criterio == "rss"
        else ["Gini_izq", "Gini_der", "Gini_ponderado"])
    return pd.DataFrame(filas, columns=columnas)


def mejor_corte(X, y, min_hoja=10, criterio="rss"):
    """Mejor corte de cada predictora de X, comparado con no dividir.

    Para cada predictora reporta el número de cortes admisibles, el mejor corte
    y su criterio (RSS total o Gini ponderado). La fila "sin dividir" tiene el
    criterio del grupo completo: el RSS al predecir con la media o el Gini.
    Ordenada de menor a mayor criterio: la primera fila es la división que elige
    el árbol en este nodo.
    """
    nombre = "RSS" if criterio == "rss" else "Gini"
    filas = {"sin dividir": {"cortes_admisibles": 0, "mejor_corte": np.nan,
                             nombre: {"rss": rss, "gini": gini}[criterio](y)}}
    for v in X.columns:
        cortes = evaluar_cortes(X[v], y, min_hoja, criterio)
        if cortes.empty:
            continue
        mejor = cortes.iloc[cortes.iloc[:, -1].argmin()]
        filas[v] = {"cortes_admisibles": len(cortes), "mejor_corte": mejor["corte"],
                    nombre: mejor.iloc[-1]}
    return pd.DataFrame.from_dict(filas, orient="index").sort_values(nombre)


def graficar_cortes(X, y, min_hoja=10, criterio="rss"):
    """Grafica, para cada predictora de X, el criterio de cada corte admisible
    (RSS total o Gini ponderado) frente al punto de corte. La línea punteada es
    el criterio sin dividir y la discontinua, el mejor corte de la predictora.
    Devuelve un diccionario {predictora: tabla de evaluar_cortes}.
    """
    tablas = {v: evaluar_cortes(X[v], y, min_hoja, criterio) for v in X.columns}
    sin_dividir = {"rss": rss, "gini": gini}[criterio](y)
    _, ejes = plt.subplots(1, len(tablas), sharey=True, figsize=(5 * len(tablas), 3.8),
                           squeeze=False)
    for ax, (v, t) in zip(ejes[0], tablas.items()):
        criterio_corte = t.iloc[:, -1]
        mejor = t["corte"].iloc[criterio_corte.argmin()]
        ax.plot(t["corte"], criterio_corte, marker="o", markersize=3, linewidth=1.5)
        ax.axhline(sin_dividir, color="gray", linestyle=":")
        ax.axvline(mejor, color="gray", linestyle="--")
        # Rótulos bajo la línea de no dividir, donde la curva casi nunca llega.
        ax.annotate(f"mejor corte: {mejor:g}", (mejor, sin_dividir), xytext=(5, -5),
                    textcoords="offset points", va="top", color="dimgray")
        ax.annotate("sin dividir", (1, sin_dividir), xycoords=("axes fraction", "data"),
                    xytext=(-3, 3), textcoords="offset points", ha="right", color="dimgray")
        ax.set_xlabel(f"Punto de corte en {v}")
    minimo = min(t.iloc[:, -1].min() for t in tablas.values())
    ejes[0][0].set_ylim(top=sin_dividir + 0.08 * (sin_dividir - minimo))
    ejes[0][0].set_ylabel("RSS total" if criterio == "rss" else "Gini ponderado")
    plt.tight_layout()
    plt.show()
    return tablas


# ---------------------------------------------------------------------
#                       Lectura del árbol ajustado
# ---------------------------------------------------------------------
def graficar_respuesta(X, y):
    """Diagrama de dispersión de la respuesta y frente a cada predictora de X."""
    _, ejes = plt.subplots(1, X.shape[1], sharey=True, figsize=(5 * X.shape[1], 3.8),
                           squeeze=False)
    for ax, v in zip(ejes[0], X.columns):
        ax.scatter(X[v], y, s=14, alpha=0.6)
        ax.set_xlabel(v)
    ejes[0][0].set_ylabel(y.name)
    plt.tight_layout()
    plt.show()


def graficar_arbol(modelo, decimales=3):
    """Dibuja el árbol ajustado con plot_tree. En cada nodo aparecen la regla (a
    la izquierda, True, van los valores <= corte), la impureza (squared_error =
    RSS / n, o gini), samples (observaciones de entrenamiento) y value: la media
    de la respuesta o el número de observaciones de cada clase, en el orden de
    modelo.classes_."""
    clases = [str(c) for c in modelo.classes_] if is_classifier(modelo) else None
    # Tamaño según las hojas y la profundidad, para que el texto se lea en árboles pequeños.
    ancho = min(16, max(5, 2.1 * modelo.get_n_leaves()))
    plt.figure(figsize=(ancho, 1.2 * (modelo.get_depth() + 1) + 0.4))
    plot_tree(modelo, feature_names=list(modelo.feature_names_in_), class_names=clases,
              precision=decimales, fontsize=11, rounded=True)
    plt.tight_layout()
    plt.show()


def tabla_hojas(modelo):
    """Una fila por hoja del árbol ajustado: la regla que la define (las
    condiciones desde la raíz), el número de observaciones de entrenamiento y la
    predicción. En regresión la predicción es la media de la respuesta en la
    hoja; en clasificación, la proporción de cada clase y la clase mayoritaria.
    El índice es el número de nodo que devuelve modelo.apply(X).
    """
    arbol = modelo.tree_
    reglas = _reglas(modelo)
    hojas = np.flatnonzero(arbol.children_left < 0)
    tabla = pd.DataFrame({"regla": [" y ".join(reglas[h]) for h in hojas],
                          "n": arbol.n_node_samples[hojas]},
                         index=pd.Index(hojas, name="hoja"))
    valores = arbol.value[hojas, 0, :]
    if is_classifier(modelo):
        proporciones = valores / valores.sum(axis=1, keepdims=True)
        for k, clase in enumerate(modelo.classes_):
            tabla[f"p_{clase}"] = proporciones[:, k]
        tabla["clase"] = modelo.classes_[proporciones.argmax(axis=1)]
    else:
        tabla["prediccion"] = valores[:, 0]
    return tabla


def predecir_con_ruta(modelo, X_nuevos):
    """Predicción de cada fila de X_nuevos junto con la hoja a la que llega y la
    regla que recorre hasta ella: la explicación de la predicción en un árbol.
    Las predictoras que no aparecen en la regla no influyen en esa predicción.
    La observación no se agrega a la hoja: la predicción es la de tabla_hojas.
    """
    hojas = tabla_hojas(modelo)
    return X_nuevos.assign(hoja=modelo.apply(X_nuevos)).join(hojas, on="hoja")


def graficar_regiones(modelo, X, color=None, etiquetas=False):
    """Partición del plano de las dos predictoras de X que define el árbol.

    Cada corte se dibuja solo dentro de la región que divide; los puntos son las
    observaciones de X y cada rectángulo muestra la predicción de su hoja (la
    media en regresión, la clase en clasificación) y su número de observaciones
    de entrenamiento. color: opcional, la respuesta de X para colorear los
    puntos (numérica: escala de un tono; categórica: un color y un marcador por
    clase). etiquetas=True rotula cada punto con su índice en X (útil con pocos
    datos). Devuelve la tabla de tabla_hojas.
    """
    if X.shape[1] != 2:
        raise ValueError("graficar_regiones requiere exactamente dos predictoras.")
    arbol = modelo.tree_
    hojas = tabla_hojas(modelo)
    v1, v2 = X.columns
    ancho, alto = np.ptp(X[v1]), np.ptp(X[v2])
    limites = ((X[v1].min() - 0.04 * ancho, X[v1].max() + 0.04 * ancho),
               (X[v2].min() - 0.04 * alto, X[v2].max() + 0.04 * alto))
    tamano = 40 if len(X) <= 30 else 14
    _, ax = plt.subplots(figsize=(7.5, 4.8))
    if color is None:
        ax.scatter(X[v1], X[v2], s=tamano, color="gray", alpha=0.6)
    elif pd.api.types.is_numeric_dtype(color):
        # Escala de un solo tono que empieza en un azul visible sobre el fondo blanco.
        azules = LinearSegmentedColormap.from_list("azules",
                                                   plt.cm.Blues(np.linspace(0.3, 1, 256)))
        puntos = ax.scatter(X[v1], X[v2], c=color, cmap=azules, s=tamano)
        plt.colorbar(puntos, ax=ax, label=color.name)
    else:
        for i, clase in enumerate(sorted(pd.unique(color))):
            m = np.asarray(color == clase)
            ax.scatter(X[v1][m], X[v2][m], s=tamano + 2, color=f"C{i}",
                       marker="o^sD"[i % 4], alpha=0.7, label=clase)
        ax.legend(title=color.name, loc="upper left", bbox_to_anchor=(1, 1))
    if etiquetas:
        for nombre, (a, b) in X[[v1, v2]].iterrows():
            ax.annotate(str(nombre), (a, b), xytext=(5, 5), textcoords="offset points")

    def recorrer(nodo, caja):
        (x0, x1), (y0, y1) = caja
        if arbol.children_left[nodo] < 0:  # hoja: se rotula en el centro de su región
            h = hojas.loc[nodo]
            texto = (f"{h['clase']}" if is_classifier(modelo) else f"{h['prediccion']:.2f}")
            angosta = (x1 - x0) < 0.12 * (limites[0][1] - limites[0][0])
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, f"{texto}\nn = {h['n']}", ha="center",
                    va="center", fontsize=8, rotation=90 if angosta else 0,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85,
                              edgecolor="none"))
            return
        corte = arbol.threshold[nodo]
        if modelo.feature_names_in_[arbol.feature[nodo]] == v1:
            ax.plot([corte, corte], [y0, y1], color="black", linewidth=1.5)
            izq, der = ((x0, corte), (y0, y1)), ((corte, x1), (y0, y1))
        else:
            ax.plot([x0, x1], [corte, corte], color="black", linewidth=1.5)
            izq, der = ((x0, x1), (y0, corte)), ((x0, x1), (corte, y1))
        recorrer(arbol.children_left[nodo], izq)
        recorrer(arbol.children_right[nodo], der)

    recorrer(0, limites)
    ax.grid(False)  # las únicas líneas del gráfico son los cortes
    ax.set(xlim=limites[0], ylim=limites[1], xlabel=v1, ylabel=v2)
    plt.tight_layout()
    plt.show()
    return hojas


# ---------------------------------------------------------------------
#                               Evaluación
# ---------------------------------------------------------------------
def errores_regresion(modelos, X_ent, y_ent, X_pru, y_pru):
    """Ajusta cada modelo del diccionario {nombre: modelo} con el entrenamiento y
    reporta su RSS, MSE y RMSE de entrenamiento y de prueba:
        RSS = sum (y_i - y_hat_i)^2, MSE = RSS / n (unidades de Y al cuadrado) y
        RMSE = raíz del MSE (unidades de Y).
    Los modelos quedan ajustados.
    """
    filas = {}
    for nombre, modelo in modelos.items():
        modelo.fit(X_ent, y_ent)
        fila = {}
        for conjunto, X, y in [("entrenamiento", X_ent, y_ent), ("prueba", X_pru, y_pru)]:
            cuadrados = (np.asarray(y) - modelo.predict(X)) ** 2
            fila[(conjunto, "RSS")] = cuadrados.sum()
            fila[(conjunto, "MSE")] = cuadrados.mean()
            fila[(conjunto, "RMSE")] = np.sqrt(cuadrados.mean())
        filas[nombre] = fila
    return pd.DataFrame.from_dict(filas, orient="index")


def curva_profundidad(modelo, X_ent, y_ent, X_pru, y_pru, profundidades, k=10):
    """Error de entrenamiento, de validación cruzada y de prueba del árbol para
    cada profundidad máxima de la lista `profundidades`.

    modelo: árbol sin ajustar; conserva sus demás parámetros (tamaños mínimos).
    La validación cruzada usa solo el entrenamiento, con k particiones
    consecutivas (KFold sin barajar), y es la que sirve para elegir la
    profundidad; el error de prueba solo se reporta. En regresión el error es el
    MSE y en clasificación la tasa de error.
    Devuelve un DataFrame (filas: profundidades) con el número de hojas, los tres
    errores y el error estándar de la validación cruzada, y grafica las tres
    curvas; la línea vertical marca la menor validación cruzada.
    """
    filas = {}
    for d in profundidades:
        arbol = clone(modelo).set_params(max_depth=d)
        vc = cross_validate(arbol, X_ent, y_ent, cv=KFold(k), scoring=_metrica(arbol))
        errores_vc = _a_error(arbol, vc["test_score"])
        arbol.fit(X_ent, y_ent)
        filas[d] = {"hojas": arbol.get_n_leaves(),
                    "entrenamiento": _error(arbol, y_ent, arbol.predict(X_ent)),
                    "validacion_cruzada": errores_vc.mean(),
                    "ee_validacion": errores_vc.std(ddof=1) / np.sqrt(k),
                    "prueba": _error(arbol, y_pru, arbol.predict(X_pru))}
    tabla = pd.DataFrame.from_dict(filas, orient="index").rename_axis("profundidad")
    curvas = tabla[["entrenamiento", "validacion_cruzada", "prueba"]].set_axis(
        ["Entrenamiento", "Validación cruzada", "Prueba"], axis=1)
    ax = curvas.plot(style=["o-", "s-", "^-"], xlabel="Profundidad máxima",
                     ylabel=_nombre_error(modelo))
    ax.axvline(tabla["validacion_cruzada"].idxmin(), color="gray", linestyle="--")
    plt.show()
    return tabla


def matriz_confusion(y, pred, positiva):
    """Matriz de confusión con la clase observada en las filas y la predicha en
    las columnas; la clase positiva va al final (por ejemplo, Bajo y Alto)."""
    clases = sorted(set(pd.unique(np.asarray(y))) | set(pd.unique(np.asarray(pred))),
                    key=lambda c: (c == positiva, str(c)))
    return pd.crosstab(pd.Series(np.asarray(y), name="Observado"),
                       pd.Series(np.asarray(pred), name="Predicho")).reindex(
        index=clases, columns=clases, fill_value=0)


def metricas_clasificacion(modelos, X_ent, y_ent, X_pru, y_pru, positiva):
    """Ajusta cada clasificador binario del diccionario {nombre: modelo} con el
    entrenamiento y lo evalúa en prueba, con `positiva` como clase positiva:
        TCC = (VP + VN) / n            tasa de clasificación correcta
        error = (FP + FN) / n = 1 - TCC
        sensibilidad = VP / (VP + FN)  positivos observados bien clasificados
        especificidad = VN / (VN + FP) negativos observados bien clasificados
    La clase predicha es la más probable (umbral 0.5 con dos clases).
    Devuelve la tabla de métricas y un diccionario {nombre: matriz de confusión}.
    Los modelos quedan ajustados.
    """
    filas, matrices = {}, {}
    for nombre, modelo in modelos.items():
        matriz = matriz_confusion(y_pru, modelo.fit(X_ent, y_ent).predict(X_pru), positiva)
        vn, fp, fn, vp = matriz.to_numpy().ravel()
        n = vn + fp + fn + vp
        filas[nombre] = {"TCC": (vp + vn) / n, "error": (fp + fn) / n,
                         "sensibilidad": vp / (vp + fn), "especificidad": vn / (vn + fp)}
        matrices[nombre] = matriz
    return pd.DataFrame.from_dict(filas, orient="index"), matrices
