library(car)
library(klaR)
library(leaps)
source("functions.R")
source("funciones_cv.R")

data <- read.csv("data/DatosSemana1.csv")

data$Estrato <- factor(data$ESTRAT)

mod1 <- lm(Y~PuntMat + Estrato, data = data)

summary(mod1)

data$Estrato <- relevel(data$Estrato, ref = "1")

mod1 <- lm(Y~PuntMat + Estrato, data = data)

summary(mod1)



mod_red <- lm(Y~PuntMat, data = data)

anova(mod_red, mod1)


mod_int <- lm(Y~PuntMat*Estrato, data = data)
anova(mod1, mod_int)


m2 <- lm(Y~PuntMat+Edad, data = data)
myCoefficients(m2, data[, c("Y", "PuntMat", "Edad")])


cor(data[, c("PuntMat", "Edad")])
vif(m2)

cond.index(Y~PuntMat+Edad, data = data)



m_comp <- lm(Y~PuntMat+Edad+Estrato, data = data)

tab <- myAllRegTable(m_comp)

n_estr <- sapply(strsplit(tab$Variables_in_model, " "),
                 function(v) sum(grepl("Estrato", v)))

tab[n_estr %in% c(0,6),]


formulas <- list(
  X1 = Y~PuntMat,
  X2 = Y~Edad,
  X1_X2 = Y~PuntMat + Edad,
  W = Y~Estrato,
  X1_W = Y~PuntMat + Estrato,
  X2_W = Y~Edad + Estrato,
  X1_X2_W = Y~PuntMat + Edad + Estrato
)

part <- crear_particiones(data, semilla = 123)
validacion_cruzada(formulas, part$loocv)

