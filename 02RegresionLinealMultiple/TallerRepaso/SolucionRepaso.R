library(car)    
library(klaR)   
library(leaps) 

source("02RegresionLinealMultiple/TallerRepaso/functions.R")

data <- read.csv("02RegresionLinealMultiple/TallerRepaso/fisio.csv")

modelo <- lm(Y~X1+X2+X3+X4+X5+X6, data = data)

summary(modelo)

myAllRegTable(modelo)
myCp_criterion(modelo)
myR2_criterion(modelo)
myAdj_R2_criterion(modelo)


residuales_press <- residuals(modelo)/(1-hatvalues(modelo))
estadistico_press <- sum(residuales_press**2)

estadistico_press


modelo <- lm(Y~X2+X3+X5+X6, data = data)


residuales_press <- residuals(modelo)/(1-hatvalues(modelo))
estadistico_press <- sum(residuales_press**2)

estadistico_press

modelo <- lm(Y~X1+X2+X3+X4+X5+X6, data = data)

myBackward(data)
myStepwise(modelo, alpha.to.enter = 0.05, alpha.to.leave = 0.05)
myStepwise(modelo, alpha.to.enter = 0.05, alpha.to.leave = 1) ##Forward



vif(modelo)
cond.index(Y~X1+X2+X3+X4+X5+X6, data = data)
myCollinDiag(modelo)

cor(data[, -1])
