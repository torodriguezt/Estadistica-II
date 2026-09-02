data <- read.csv("02RegresionLinealMultiple/05QuintoTaller/data/imcv_modelo_limpio.csv")

source("02RegresionLinealMultiple/05QuintoTaller/functions.R")

modelo <- lm(IMCV~Ingresos + Salud + Trabajo + Capital, data = data)

data_backward <- data[, -6]

myBackward(data_backward)
myStepwise(modelo, alpha.to.enter = 0.05, alpha.to.leave = 0.05)


modelo_primero <- lm(IMCV~Ingresos, data = data)
modelo_tercero <- lm(IMCV~Salud + Ingresos, data = data)
modelo_cuarto <- lm(IMCV~Capital + Ingresos, data = data)
modelo_quinto <- lm(IMCV~Trabajo + Ingresos, data = data)

anova(modelo_primero, modelo_tercero)
anova(modelo_primero, modelo_cuarto)


##Segundo Punto

data_segundo <- data[c(1, 3, 4, 5, 6)] 
data_segundo$NivelIngresos <- as.factor(data_segundo$NivelIngresos)

data_segundo$NivelIngresos <- relevel(data_segundo$NivelIngresos, ref = "Bajo")

modelo_ind <- lm(IMCV~Salud + NivelIngresos + Salud:NivelIngresos, data = data_segundo)

library(car)

linearHypothesis(modelo_ind, "NivelIngresosMedio = 0")
linearHypothesis(modelo_ind, "NivelIngresosMedio - NivelIngresosAlto = 0")
