data <- read.csv("data/nacimientos_simplificado.csv")

data$REGIMEN <- factor(data$REGIMEN, levels = c("Contributivo", "Subsidiado", 
                                                "NoAsegurado"))

modelo <- glm(CONSULTAS~BAJO_PESO + EDAD_MADRE + GEMELAR + REGIMEN,
              data = data, family=poisson(link = "log"))

summary(modelo)


lambda <- fitted(modelo)
lambda[1:10]

conteos <- 0:19

lambda_matriz <- matrix(rep(lambda, times = length(conteos)),
                        nrow = length(lambda))

conteos_matriz <- matrix(rep(conteos, each = length(lambda)),
                         nrow = length(lambda))


esperadas <- colSums(dpois(conteos_matriz, lambda = lambda_matriz))
esperadas <- c(esperadas, sum(ppois(19, lambda, lower.tail = FALSE)))

frecuencias <- data.frame(
  conteo = 0:20,
  observadas = tabulate(pmin(data$CONSULTAS, 20)+1, nbins = 21),
  esperadas = esperadas
)

centros <- barplot(frecuencias$observadas, names.arg = c(0:19, "20+"))

modelo_nulo <- glm(CONSULTAS~1, data = data,
                   family = poisson(link = "log"))

anova(modelo_nulo, modelo, test = "Chisq")


summary(modelo)
