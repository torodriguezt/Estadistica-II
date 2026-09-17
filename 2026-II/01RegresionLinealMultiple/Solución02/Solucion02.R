data <- read.csv("01RegresionLinealMultiple/02SegundoTaller/data/imcv_modelo_limpio.csv")  # Modifique segun donde se encuentre

head(data)

# 1) Modelo, recta ajustada, valores ajustados, residuales y sigma^2
design_matrix <- model.matrix(IMCV ~ ., data = data)
y <- data$IMCV

n <- nrow(design_matrix)
p <- ncol(design_matrix)
k <- p - 1

# Calculamos el estimador
XtX_inv <- solve(t(design_matrix) %*% design_matrix)
beta_hat <- XtX_inv %*% t(design_matrix) %*% y
beta_hat

# Matriz hat
H <- design_matrix %*% XtX_inv %*% t(design_matrix)

# Valores ajustados
y_hat <- H %*% y
head(y_hat)

# Residuales
residual <- (diag(n) - H) %*% y
head(residual)

# Estimacion de la varianza
sigma_hat <- sum(residual^2) / (n - p)
sigma_hat

# Verificacion con lm()
modelo <- lm(IMCV ~ ., data = data)

modelo$coefficients
head(modelo$fitted.values)
head(modelo$residuals)
summary(modelo)$sigma^2

# 2) Tabla ANOVA y significancia global de la regresion
SST <- sum((y - mean(y))^2)
SSE <- sum(residual^2)
SSR <- sum((y_hat - mean(y))^2)

c(SST = SST, SSR = SSR, SSE = SSE, SSR + SSE)

# Cuadrados medios y estadistico F0
MSR <- SSR / k
MSE <- SSE / (n - p)
F0 <- MSR / MSE

tabla_anova <- data.frame(
  Fuente = c("Regresion", "Error", "Total"),
  SS     = c(SSR, SSE, SST),
  gl     = c(k, n - p, n - 1),
  MS     = c(MSR, MSE, NA),
  F0     = c(F0, NA, NA)
)
tabla_anova

# Region de rechazo y valor-p
cuantil_f <- qf(0.95, k, n - p)
cuantil_f
pf(F0, k, n - p, lower.tail = FALSE)

# Misma prueba por sumas de cuadrados extra
reduced_model <- lm(IMCV ~ 1, data = data)
full_model <- lm(IMCV ~ ., data = data)

SSE_R <- sum(reduced_model$residuals^2)
SSE_F <- sum(full_model$residuals^2)

SS_extra <- SSE_R - SSE_F
gl_extra <- reduced_model$df.residual - full_model$df.residual

F0_extra <- (SS_extra / gl_extra) / (SSE_F / (n - p))
c(SS_extra = SS_extra, gl = gl_extra, F0 = F0_extra)

pf(F0_extra, gl_extra, n - p, lower.tail = FALSE)

anova(reduced_model, full_model)

# 3) Significancia individual con pruebas t e intervalos de confianza
var_beta <- sigma_hat * XtX_inv
ee_beta <- sqrt(diag(var_beta))

estadistico_t <- beta_hat / ee_beta
valor_p <- 2 * pt(abs(estadistico_t), n - p, lower.tail = FALSE)

cuantil_t <- qt(0.975, n - p)
cuantil_t

# IC del 95%
IC <- cbind(beta_hat - cuantil_t * ee_beta,
            beta_hat + cuantil_t * ee_beta)

data.frame(Estimacion = beta_hat, ee = ee_beta, T0 = estadistico_t,
           valor_p = valor_p, LI = IC[, 1], LS = IC[, 2])

summary(modelo)
confint(modelo)

# 4) R^2, R^2_adj y comparacion con el modelo reducido
R2 <- SSR / SST
R2_adj <- 1 - (n - 1) * MSE / SST
c(R2 = R2, R2_adj = R2_adj, MSE = MSE)

summary(modelo)$r.squared
summary(modelo)$adj.r.squared

# La covariable menos significativa en 3) es Ingresos
modelo_reducido <- lm(IMCV ~ Salud + Trabajo + Capital, data = data)

MSE_reducido <- sum(modelo_reducido$residuals^2) / modelo_reducido$df.residual
R2_adj_reducido <- 1 - (n - 1) * MSE_reducido / SST

data.frame(
  Modelo  = c("Completo", "Reducido (sin Ingresos)"),
  R2      = c(R2, summary(modelo_reducido)$r.squared),
  R2_adj  = c(R2_adj, R2_adj_reducido),
  MSE     = c(MSE, MSE_reducido)
)

# 5) Sumas de cuadrados extra
modelo_B <- lm(IMCV ~ Ingresos + Capital, data = data)

SSE_B <- sum(modelo_B$residuals^2)
SS_extra <- SSE_B - SSE_F
gl_extra <- modelo_B$df.residual - full_model$df.residual

F0 <- (SS_extra / gl_extra) / MSE
c(SS_extra = SS_extra, gl = gl_extra, F0 = F0)

qf(0.95, gl_extra, n - p)
pf(F0, gl_extra, n - p, lower.tail = FALSE)

anova(modelo_B, full_model)

# Un unico coeficiente
modelo_sin_trabajo <- lm(IMCV ~ Ingresos + Salud + Capital, data = data)

SS_extra_trabajo <- sum(modelo_sin_trabajo$residuals^2) - SSE_F
F0_trabajo <- (SS_extra_trabajo / 1) / MSE

c(SS_extra = SS_extra_trabajo, F0 = F0_trabajo)
pf(F0_trabajo, 1, n - p, lower.tail = FALSE)

# Coincide con la prueba t del punto 3
estadistico_t["Trabajo", ]^2
valor_p["Trabajo", ]

anova(modelo_sin_trabajo, full_model)
