data <- read.csv("imcv_modelo_limpio.csv")

modelo <- lm(IMCV~Ingresos+Salud+Trabajo+Capital, data = data)

plot(modelo)

shapiro.test(residuals(modelo))

res_student <- rstudent(modelo)

plot(res_student, ylab = "Residuales studentizado", xlab = "Observación", pch = 19)

abline(h = 0, lty = 2)
abline(h = c(-3, 3), lty = 2)

which(abs(res_student) > 3)
res_student[which(abs(res_student) > 3)]

x01 <- c(1, 1.6921, 3.3214, 0.6125, 6.1313)
x02 <- c(1, 3.6, 4.3, 0.4, 3.3)

X <- model.matrix(modelo)
Hat_values <- hatvalues(modelo)

h01 <- t(x01) %*% solve(t(X) %*% X) %*% x01
h02 <- t(x02) %*% solve(t(X) %*% X) %*% x02

ifelse(h01 < max(Hat_values), "Pertenece a la región", "No pertenece a la región")
ifelse(h02 < max(Hat_values), "Pertenece a la región", "No pertenece a la región")


predict(modelo,
        newdata = data.frame(
          Ingresos = 1.6921,
          Salud = 3.3214,
          Trabajo = 0.6125,
          Capital = 6.1313
        ),
        interval = "prediction")
