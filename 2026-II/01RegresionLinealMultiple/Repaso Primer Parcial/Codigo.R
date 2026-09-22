data <- read.table("fisio.txt", header = TRUE)

modelo <- lm(Y~X1+X2+X3+X4+X5+X6, data = data)

summary(modelo)

modelo_nulo <- lm(Y~1, data = data)
anova(modelo_nulo, modelo) # evalua prueba de hipotesis de significancia global

mod_sin_pulo <- lm(Y~X1+X2+X3, data = data)
anova(mod_sin_pulo, modelo)


mod_completo2 <- lm(Y~X1+X2+X3+X5+X6, data = data)
mod_reducido2 <- lm(Y~X3+X5+X6, data = data)

anova(mod_reducido2, mod_completo2)


par(mfrow=c(2,2))

plot(modelo)

shapiro.test(residuals(modelo))

ti <- rstudent(modelo)

which(abs(ti) > 3)

plot(ti, pch = 19, ylim = c(-3.5, 3.5))
abline(h = c(-3, 3), col = "red")


Hat_values <- hatvalues(modelo)
X <- model.matrix(modelo)
XtXi <- solve(t(X)%*%X)

x01 <- c(1, 48, 77, 11, 54, 169, 173)
x02 <- c(1, 38, 50, 9, 40, 140, 155)

h01 <- t(x01)%*%XtXi%*%x01
h02 <- t(x02)%*%XtXi%*%x02

ifelse(h01 < max(Hat_values), "Pertenece a la región", "No pertenece a la región")
ifelse(h02 < max(Hat_values), "Pertenece a la región", "No pertenece a la región")


x0 <- data.frame(X1 = 48, X2 = 77, X3 = 11, X4 = 54, X5 = 169, X6 = 173)

predict(modelo, newdata = x0, interval = "confidence", level = 0.95)

