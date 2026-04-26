library(car)

data <- read.csv("data/imcv_modelo_limpio.csv")

fm <- lm(IMCV ~ Ingresos + Salud + Trabajo + Capital, data = data)
summary(fm)

n   <- nrow(data)
p   <- length(coef(fm))
X   <- model.matrix(fm)
MSE <- sum(resid(fm)^2) / (n - p)
SSE_fm <- sum(resid(fm)^2)
SST    <- sum((data$IMCV - mean(data$IMCV))^2)

# =============================================================
# Punto 1 - Intervalos de confianza para los beta_j
# =============================================================

confint(fm, level = 0.95)

# A mano
beta   <- solve(t(X) %*% X) %*% t(X) %*% data$IMCV
cjj    <- diag(solve(t(X) %*% X))
ee     <- sqrt(MSE * cjj)
t_crit <- qt(0.975, df = n - p)

data.frame(
  parametro = names(coef(fm)),
  estimado  = as.numeric(beta),
  LI        = as.numeric(beta) - t_crit * ee,
  LS        = as.numeric(beta) + t_crit * ee
)

# =============================================================
# Punto 2 - Verificacion F_{j,0} = T_{j,0}^2 para Salud
# =============================================================

# T a mano
beta_sal  <- coef(fm)["Salud"]
ee_sal    <- sqrt(MSE * cjj["Salud"])
T_mano    <- beta_sal / ee_sal
pv_T_mano <- 2 * pt(abs(T_mano), df = n - p, lower.tail = FALSE)

cat("T_{j,0}   =", round(T_mano, 6), "\n")
cat("p-valor t =", round(pv_T_mano, 6), "\n")

# F via SSextra a mano
X_rm2    <- model.matrix(~ Ingresos + Trabajo + Capital, data = data)
beta_rm2 <- solve(t(X_rm2) %*% X_rm2) %*% t(X_rm2) %*% data$IMCV
SSE_rm2  <- sum((data$IMCV - X_rm2 %*% beta_rm2)^2)
F_mano   <- (SSE_rm2 - SSE_fm) / MSE
pv_F_mano <- pf(F_mano, df1 = 1, df2 = n - p, lower.tail = FALSE)

cat("T^2       =", round(T_mano^2, 6), "\n")
cat("F_{j,0}   =", round(F_mano, 6), "\n")
cat("p-valor F =", round(pv_F_mano, 6), "\n")

# =============================================================
# Punto 3 - SSextra para {beta_Trabajo, beta_Capital}
# =============================================================

# H0: beta_Trabajo = beta_Capital = 0

# A mano
X_rm3    <- model.matrix(~ Ingresos + Salud, data = data)
beta_rm3 <- solve(t(X_rm3) %*% X_rm3) %*% t(X_rm3) %*% data$IMCV
SSE_rm3  <- sum((data$IMCV - X_rm3 %*% beta_rm3)^2)
F0_3     <- ((SSE_rm3 - SSE_fm) / 2) / MSE
pv_3     <- pf(F0_3, df1 = 2, df2 = n - p, lower.tail = FALSE)

cat("F_0     =", round(F0_3, 4), "\n")
cat("p-valor =", round(pv_3, 6), "\n")

# Verificacion
rm_3 <- lm(IMCV ~ Ingresos + Salud, data = data)
anova(rm_3, fm)

# =============================================================
# Punto 4 - Metodo lineal general
# =============================================================

# H0: beta_Ing = beta_Sal  y  beta_Tra + beta_Cap = 1
# L*beta = c

L     <- matrix(c(0, 1, -1, 0, 0,
                  0, 0,  0, 1, 1), nrow = 2, byrow = TRUE)
c_vec <- c(0, 1)

# A mano via modelo reducido
# De H0: beta_Ing = beta_Sal = g1, beta_Cap = 1 - beta_Tra = 1 - g2
# => Y - Capital = beta0 + g1*(Ing+Sal) + g2*(Tra-Cap) + eps

Y_star <- data$IMCV - data$Capital
Z1     <- data$Ingresos + data$Salud
Z2     <- data$Trabajo  - data$Capital

X_rm4    <- cbind(1, Z1, Z2)
beta_rm4 <- solve(t(X_rm4) %*% X_rm4) %*% t(X_rm4) %*% Y_star
SSE_rm4  <- sum((Y_star - X_rm4 %*% beta_rm4)^2)
F0_4     <- ((SSE_rm4 - SSE_fm) / 2) / MSE
pv_4     <- pf(F0_4, df1 = 2, df2 = n - p, lower.tail = FALSE)

cat("F_0     =", round(F0_4, 4), "\n")
cat("p-valor =", round(pv_4, 6), "\n")

# Verificacion
linearHypothesis(fm, hypothesis.matrix = L, rhs = c_vec)

# =============================================================
# Punto 5 - Comparacion R2 vs R2_adj
# =============================================================

R2_fm    <- 1 - SSE_fm / SST
R2adj_fm <- 1 - (n - 1) * MSE / SST

# Modelo sin Ingresos (mayor p-valor marginal)
X_rm5    <- model.matrix(~ Salud + Trabajo + Capital, data = data)
beta_rm5 <- solve(t(X_rm5) %*% X_rm5) %*% t(X_rm5) %*% data$IMCV
SSE_rm5  <- sum((data$IMCV - X_rm5 %*% beta_rm5)^2)
MSE_rm5  <- SSE_rm5 / (n - (p - 1))
R2_rm5    <- 1 - SSE_rm5 / SST
R2adj_rm5 <- 1 - (n - 1) * MSE_rm5 / SST

data.frame(
  Modelo = c("Completo", "Sin Ingresos"),
  R2     = c(R2_fm,    R2_rm5),
  R2_adj = c(R2adj_fm, R2adj_rm5),
  MSE    = c(MSE,      MSE_rm5)
)