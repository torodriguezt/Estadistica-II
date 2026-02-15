datos <- read.csv("../data/imcv_modelo_limpio.csv")

head(datos)

#a)
design_matrix <- model.matrix(IMCV~., data = datos)

head(design_matrix)

y <- datos$IMCV
beta_hat <- solve(t(design_matrix)%*%design_matrix)%*%t(design_matrix)%*%y

beta_hat

#b)
y_hat <- design_matrix%*%beta_hat
sigma_hat <- sum((y-y_hat)**2)/(nrow(datos)-ncol(design_matrix))
sigma_hat

#c)

var_hat_beta_hat <- sigma_hat*solve((t(design_matrix)%*%design_matrix))


##Verificacion con lm()

modelo <- lm(IMCV~., data = datos)
modelo$coefficients

summary(modelo)$sigma**2

vcov(modelo)
