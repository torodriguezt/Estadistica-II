# Repositorio del curso Estadística II

En este repositorio se encuentra la información asociada al curso de *Estadística II* de la **Universidad Nacional de Colombia, sede Medellín**, para el periodo académico 2026-II. El curso tiene como objetivo general **conocer y profundizar** en los fundamentos básicos, propiedades y aplicaciones de los **modelos de regresión lineal**, la **selección y validación de modelos**, la **regresión logística**, la **regresión de Poisson**, los **árboles, bosques aleatorios y métodos de ensamble (bagging y boosting)**, y la **interpretación de modelos de caja negra (XAI)**. El curso tiene una orientación teórico-práctica.

Es de especial interés tener en cuenta la siguiente información:

**(1). Talleres prácticos:** Cada semana se ofrecerá un espacio destinado a la solución del taller práctico. El objetivo es profundizar los conocimientos adquiridos en el aula, por lo que es recomendable asistir con una base teórica decente 😄. El horario del taller es el siguiente:

- **Viernes (Virtual).** A través de Google Meet. (17:00 - 19:00)

**(2). Conocimientos previos:** Estadística I, en particular, pruebas de hipótesis e intervalos de confianza (muy importantes). Es deseable también conocimientos básicos en álgebra lineal y de programación en R o Python. Se recomienda RStudio como IDE para R y Visual Studio Code para Python. Una guía simple de instalación es la siguiente: https://www.youtube.com/watch?v=hbgzW3Cvda4

**(3). Medios de comunicación:** torodriguezt@unal.edu.co

**(4). Grabaciones:** Estarán hospedadas en Google Drive. [Enlace Drive](https://drive.google.com/drive/folders/1y4_9sj74vLJJHkjjSHci1LKpp5o3jQ0x?usp=sharing)

## Temas del curso

1. **Regresión lineal múltiple:** enfoque matricial, interpretación de coeficientes, estimación por MCO y máxima verosimilitud, propiedades de los estimadores.
2. **Inferencia en el modelo lineal:** análisis de varianza, $R^2$ y $R^2_{adj}$, pruebas e intervalos sobre coeficientes, sumas de cuadrados extra y pruebas sobre subconjuntos de coeficientes.
3. **Diagnósticos y validación:** análisis de residuales, verificación de supuestos, observaciones atípicas e influyentes, multicolinealidad.
4. **Regularización y selección de modelos:** regresión Ridge y Lasso, todas las regresiones posibles, métodos secuenciales (significancia y criterios de información) y validación cruzada.
5. **Variables indicadoras:** regresión con predictoras categóricas e interacciones.
6. **Regresión logística:** modelo Bernoulli, estimación, interpretación, inferencia, matriz de confusión y curva ROC.
7. **Regresión Poisson:** estimación, interpretación de parámetros, inferencia y predicción.
8. **Árboles:** árboles de regresión y de clasificación, estratificación del espacio de predictoras, ventajas y desventajas.
9. **Métodos de ensamble:** bagging, random forests y boosting.
10. **Interpretabilidad (XAI):** métodos de interpretación de modelos de caja negra.

## Estructura del repositorio

- `2026-II/` — material del semestre actual.
- `2026-I/` — material de semestres anteriores (referencia).
- `Recursos/` — material adicional.

Cada carpeta principal de un tema (Ej: `01RegresionLinealMultiple`) contiene las carpetas de **Taller** y **Solución**. La carpeta de solución, que es la que más interesa, contiene:

| Archivo          | Descripción                                                                   |
| ---------------- | ------------------------------------------------------------------------------ |
| `Solucion.R`   | Código usado en el taller sin personalizaciones gráficas.                    |
| `Solucion.Rmd` | Código con personalizaciones gráficas, tablas y contenido para crear el PDF. |
| `Solucion.pdf` | Documento final con el código base de la sesión y sus gráficas.             |
| `data/`        | Base de datos usada en el taller.                                              |

La carpeta `Recursos/` contiene material complementario: scripts de apoyo sobre varianza, multicolinealidad, predicción, regresión logística y Poisson, y una guía de comandos de R y de Git (esta última es completamente opcional y no es necesaria para el curso, pero facilitará su interacción con el repositorio).
