data <- read.csv("../01PrimerTaller/data/nacimientos_simplificado.csv",
                 stringsAsFactors = TRUE,
                 encoding = "UTF-8")

data$REGIMEN <- relevel(factor(data$REGIMEN), ref = "Contributivo")

modelo <- glm(BAJO_PESO ~ EDAD_MADRE + CONSULTAS + GEMELAR + REGIMEN,
              data = data, family = binomial(link = "logit"))

summary(modelo)

exp(coef(modelo))
