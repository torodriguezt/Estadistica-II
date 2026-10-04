library(car)          # vif()
library(klaR)         # cond.index()
library(glmnetUtils)  # Ridge y Lasso con fórmula
library(leaps)        # Todas las regresiones posibles
source("functions.R")

data <- read.csv("data/imcv_modelo_limpio.csv")

# -----------------------------
#          PRIMER PUNTO
# -----------------------------
# Grafico de dispersion por pares
pairs(data)

modelo <- lm(IMCV ~ Ingresos + Salud + Trabajo + Capital, data = data)
summary(modelo)    # Ingresos negativo y no significativo; F global significativa

# Matriz de correlacion
X <- model.matrix(modelo)
cor(X[, -1])

# -----------------------------
#          SEGUNDO PUNTO
# -----------------------------
# Indices de condicion; el mayor es la raiz del numero de condicion
cond.index(IMCV ~ Ingresos + Salud + Trabajo + Capital, data = data)

# VIF
vif(modelo)

# -----------------------------
#          TERCER PUNTO
# -----------------------------
lambda.grid <- exp(seq(log(5), log(1e-4), length.out = 100))

set.seed(50)
# nfolds = 3 activa la CV para escoger lambda
ridgecv <- cv.glmnet(IMCV ~ ., data = data, alpha = 0, lambda = lambda.grid,
                     nfolds = 3, type.measure = "mse")

ridgecv$lambda.min
coef(ridgecv, s = "lambda.min")
plot(ridgecv, sign.lambda = 1)

# Trayectorias de los coeficientes
plot(ridgecv$glmnet.fit, xvar = "lambda", sign.lambda = 1)

# Comparación con mínimos cuadrados
cbind(MCO = coef(modelo), Ridge = as.numeric(coef(ridgecv, s = "lambda.min")))

# -----------------------------
#          CUARTO PUNTO
# -----------------------------
set.seed(50)
# nfolds = 3 activa la CV para escoger lambda
lassocv <- cv.glmnet(IMCV ~ ., data = data, alpha = 1, lambda = lambda.grid,
                     nfolds = 3, type.measure = "mse")

lassocv$lambda.min
coef(lassocv, s = "lambda.min")
plot(lassocv, sign.lambda = 1)

# Trayectorias de los coeficientes
plot(lassocv$glmnet.fit, xvar = "lambda", sign.lambda = 1)

# Coeficientes distintos de cero
sum(coef(lassocv, s = "lambda.min")[-1] != 0)

# MSE de validación cruzada (mismas particiones)
c(Ridge = min(ridgecv$cvm), Lasso = min(lassocv$cvm))

# -----------------------------
#          QUINTO PUNTO
# -----------------------------
# Criterio R2
myR2_criterion(modelo)
# Criterio R2 ajustado
myAdj_R2_criterion(modelo)
# Criterio Cp
myCp_criterion(modelo)
# Criterio PRESS
residual_press <- residuals(modelo) / (1 - hatvalues(modelo))
Estadistica_press <- sum(residual_press^2)
Estadistica_press
# Todas las regresiones posibles (2^4 - 1 = 15 submodelos)
myAllRegTable(modelo)

# Mejor modelo: Salud + Capital (las mismas variables que conserva Lasso)
modelo_final <- lm(IMCV ~ Salud + Capital, data = data)
summary(modelo_final)
vif(modelo_final)
