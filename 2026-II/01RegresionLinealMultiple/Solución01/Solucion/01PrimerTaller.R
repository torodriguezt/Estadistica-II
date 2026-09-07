data <- read.csv("data/imcv_modelo_limpio.csv")

head(data)

matriz_diseno <- model.matrix(~Ingresos+Salud+Trabajo+Capital, data = data)
y <- data$IMCV

beta_hat <- solve(t(matriz_diseno)%*%matriz_diseno)%*%t(matriz_diseno)%*%y

y_hat <- matriz_diseno%*%beta_hat

sigma <- sum((y-y_hat)**2)/(length(y)-length(beta_hat))

var_beta <- sigma * solve(t(matriz_diseno)%*%matriz_diseno)

modelo <- lm(y ~ Ingresos + Salud + Trabajo + Capital, data = data)

vcov(modelo)

