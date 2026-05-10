datos <- read.csv("01RegresionLinealMultiple/03TercerTaller/data/imcv_modelo_limpio.csv")

modelo <- lm(IMCV~Ingresos+Salud+Trabajo+Capital, data = datos)

# -----------------------------
# Validación supuestos del modelo
# -----------------------------
# Se asume que la independencia es cierta
# Normalidad (Criterio Gráfico)
plot(modelo, 2) 
shapiro.test(modelo$residuals)

plot(modelo, 1)

modelo2 <- lm(IMCV ~ Ingresos + I(Ingresos^2) + Salud + Trabajo + Capital, data = datos)
plot(modelo2, 1)


n <- nrow(datos); p <- length(modelo$coefficients)
# Definir los valores hat (h_{i}):
hat_values <- hatvalues(modelo) # Para puntos balanceo
balanceo <- which(hat_values > ((2 * p)/n))

# Determinar puntos atípicos
estandarizados <- rstandard(modelo)
estudentizados <- rstudent(modelo)
atipicos_estandarizados <- which(abs(estandarizados) > 3)
atipicos_estudentizados <- which(abs(estudentizados) > 3)
# También se puede ver con criterio gráfico
plot(modelo, 3) # Residuales estandarizados
plot(modelo, 2) # Residuales estandarizados

cooks <- cooks.distance(modelo)
which(cooks > 1) # Verificar cooks
DFBetas <- dfbetas(modelo) # Definir
which(abs(DFBetas) > 2/sqrt(n)) # Verificar DFBETAS
DFFITS <- dffits(modelo) # Definir
which(abs(DFFITS) > (2 * sqrt(p/n))) # Verificar DFFITS
DFBetas[85, ]
DFFITS[85]
# -----------------------------
influencias <- influence.measures(modelo)
summary(influencias)
plot(modelo, 4)



X <- model.matrix(modelo)
Hat_values <- hat(X)
x01 <- c(1, 1.6921, 3.3214, 0.6125, 6.1313)
x02 <- c(1, 5, 6, 1.2, 20)
summary(X)
# -----------------------------
ifelse(t(x01)%*%solve(t(X)%*%X)%*%x01 < max(Hat_values), "Pertence a la region de diseno", "No pertenece")
ifelse(t(x02)%*%solve(t(X)%*%X)%*%x02 < max(Hat_values), "Pertence a la region de diseno", "No pertenece")
# -----------------------------
predict(modelo, newdata = data.frame(Ingresos = 1.6921, 
                                     Salud = 3.3214, 
                                     Capital = 0.6125,
                                     Trabajo = 6.1313), interval = "prediction")

