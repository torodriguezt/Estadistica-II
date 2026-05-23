data <- read.csv("01RegresionLinealMultiple/02SegundoTaller/data/imcv_modelo_limpio.csv")

design_matrix <- model.matrix(IMCV ~., data = data)

#a)

y <- data$IMCV
beta_hat <- solve(t(design_matrix)%*%design_matrix)%*%t(design_matrix)%*%y

y_hat <- design_matrix %*% beta_hat

sigma_hat <- sum((y - y_hat)**2)/(nrow(design_matrix) - ncol(design_matrix))


residuales <- y - y_hat

#b)

reduced_model <- lm(IMCV~1, data = data)
full_model <- lm(IMCV~., data = data)

SSE_R <- sum(reduced_model$residuals)**2
SSE_F <- sum(full_model$residuals)**2

SSR <- SSE_R-SSE_F

numerador_f <- SSR/(ncol(design_matrix)-1)
denominador_f <- SSE_F/(nrow(design_matrix) - ncol(design_matrix))

estadistico_f <- numerador_f/denominador_f
cuantil_f <- qf(0.95, ncol(design_matrix) - 1,(nrow(design_matrix)) - ncol(design_matrix))

pf(estadistico_f, ncol(design_matrix) - 1,(nrow(design_matrix)) - ncol(design_matrix), lower.tail = FALSE)


L <- matrix(c(0, 1, 0, 0, 0,
              0, 0, 1, 0, 0,
              0, 0, 0, 1, 0,
              0, 0, 0, 0, 1), nrow = 4, byrow =  TRUE)

XtX_inv <- solve(t(design_matrix)%*%design_matrix)

F_stat <- t(L %*% beta_hat) %*% 
  solve(L %*% XtX_inv %*% t(L)) %*%
  (L %*% beta_hat) / (4 * sigma_hat)

pf(F_stat, 4, nrow(design_matrix) - ncol(design_matrix), lower.tail = FALSE)

anova(reduced_model, full_model)



#c

summary(full_model)

confint(full_model)

#d)


L <- matrix(c(0, 1, -1, 0, 0,
              0, 1, 0, -1, 0)
            , nrow = 2, byrow =  TRUE)

XtX_inv <- solve(t(design_matrix)%*%design_matrix)

F_stat <- t(L %*% beta_hat) %*% 
  solve(L %*% XtX_inv %*% t(L)) %*%
  (L %*% beta_hat) / (2 * sigma_hat)

pf(F_stat, 2, nrow(design_matrix) - ncol(design_matrix), lower.tail = FALSE)

data$restriccion <- data$Ingresos + data$Salud + data$Trabajo

restricted_model <- lm(IMCV ~ restriccion + Capital, data = data)
anova(restricted_model, full_model)