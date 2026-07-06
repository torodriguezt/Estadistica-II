# -----------------------------
#          PRIMER PUNTO
# -----------------------------
# Lectura información
datos <- read.csv("02RegresionLinealMultiple/04CuartoTaller/data/imcv_modelo_limpio.csv")
datos <- datos[, c("IMCV", "Ingresos", "Salud", "Trabajo", "Capital")]

# Ajustar el modelo de regresión
Y  <- datos$IMCV
X1 <- datos$Ingresos; X2 <- datos$Salud
X3 <- datos$Trabajo;  X4 <- datos$Capital

modelo <- lm(Y ~ X1 + X2 + X3 + X4)

summary(modelo) # Estimaciones y significancia individual
anova(modelo)   # Significancia global

# Matriz de correlaciones entre predictores (indicios de multicolinealidad)
cor(datos[, -1])

# -----------------------------
#          SEGUNDO PUNTO
# -----------------------------
library(car)    # vif()
library(klaR)   # cond.index()
library(leaps)  # regsubsets()
source("02RegresionLinealMultiple/04CuartoTaller/functions.R")

# Gráfico de dispersión por pares
pairs(datos[, -1])

# VIF
vif(modelo)

# Valores propios y número de condición
X <- model.matrix(modelo)
cor(X[, -1])                                      # Matriz de correlación predictores
df_modelo <- data.frame(Y, X1, X2, X3, X4)
cond.index(Y ~ X1 + X2 + X3 + X4, data = df_modelo)             # Número de condición (klaR)

# Índices de condición y proporciones de varianza (Belsley)
myCollinDiag(modelo)

# -----------------------------
#          TERCER PUNTO
# -----------------------------
# Todas las regresiones posibles
myAllRegTable(modelo)

# Criterios gráficos
myR2_criterion(modelo)     # Criterio R²
myAdj_R2_criterion(modelo) # Criterio R² ajustado
myCp_criterion(modelo)     # Criterio Cp

# Criterio PRESS
residual_press     <- residuals(modelo) / (1 - hatvalues(modelo))
Estadistica_press  <- sum(residual_press^2)
Estadistica_press

# Visualización general
modelos <- olsrr::ols_step_all_possible(modelo)
plot(modelos)