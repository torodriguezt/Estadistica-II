data <- read.csv("data/nacimientos_simplificado.csv")

data$REGIMEN <- relevel(factor(data$REGIMEN), ref = "Contributivo")

modelo <- glm(NUM_HIJOS ~ ., data = data, family = poisson(link = "log"))

summary(modelo)

modelo_reducido <- glm(NUM_HIJOS ~ 1,  data = data, family = poisson(link = "log"))

anova(modelo_reducido, modelo, test = "Chisq")

#1 -pchisq(modelo$null.deviance - modelo$deviance, df = modelo$df.null - modelo$df.residual)

