# ============================================================
#   Visualización: ¿Qué es la multicolinealidad?
# ============================================================
library(ggplot2)
library(gridExtra)
library(MASS)   # mvrnorm

set.seed(42)
n <- 120

# --- Escenario 1: sin multicolinealidad ---
datos_ok <- as.data.frame(mvrnorm(n, mu = c(0, 0),
                                  Sigma = matrix(c(1, 0.05, 0.05, 1), 2)))
colnames(datos_ok) <- c("X1", "X2")

# --- Escenario 2: con multicolinealidad ---
datos_mc <- as.data.frame(mvrnorm(n, mu = c(0, 0),
                                  Sigma = matrix(c(1, 0.97, 0.97, 1), 2)))
colnames(datos_mc) <- c("X1", "X2")

# --- Colores neutros ---
col_ok <- "#1D9E75"
col_mc <- "#D85A30"

p1 <- ggplot(datos_ok, aes(X1, X2)) +
  geom_point(color = col_ok, alpha = 0.6, size = 1.8) +
  geom_smooth(method = "lm", se = FALSE,
              color = col_ok, linewidth = 0.8, linetype = "dashed") +
  labs(
    title    = "Sin multicolinealidad",
    subtitle = expression(paste("cor(X"[1]*", X"[2]*") = 0.05")),
    x = expression(X[1]), y = expression(X[2])
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "gray40", size = 11),
    panel.grid.minor = element_blank()
  )

p2 <- ggplot(datos_mc, aes(X1, X2)) +
  geom_point(color = col_mc, alpha = 0.6, size = 1.8) +
  geom_smooth(method = "lm", se = FALSE,
              color = col_mc, linewidth = 0.8, linetype = "dashed") +
  labs(
    title    = "Con multicolinealidad",
    subtitle = expression(paste("cor(X"[1]*", X"[2]*") = 0.97")),
    x = expression(X[1]), y = expression(X[2])
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "gray40", size = 11),
    panel.grid.minor = element_blank()
  )

# --- Efecto sobre los coeficientes: simulación ---
B <- 300   # repeticiones
b1_ok <- b1_mc <- numeric(B)

for (i in seq_len(B)) {
  d_ok <- as.data.frame(mvrnorm(n, c(0,0),
                                matrix(c(1, 0.05, 0.05, 1), 2)))
  d_mc <- as.data.frame(mvrnorm(n, c(0,0),
                                matrix(c(1, 0.97, 0.97, 1), 2)))
  Y_ok <- d_ok[,1] + d_ok[,2] + rnorm(n, sd = 0.5)
  Y_mc <- d_mc[,1] + d_mc[,2] + rnorm(n, sd = 0.5)
  b1_ok[i] <- coef(lm(Y_ok ~ d_ok[,1] + d_ok[,2]))[2]
  b1_mc[i] <- coef(lm(Y_mc ~ d_mc[,1] + d_mc[,2]))[2]
}

df_coef <- data.frame(
  beta  = c(b1_ok, b1_mc),
  grupo = rep(c("Sin multicolinealidad", "Con multicolinealidad"), each = B)
)
df_coef$grupo <- factor(df_coef$grupo,
                        levels = c("Sin multicolinealidad", "Con multicolinealidad"))

p3 <- ggplot(df_coef, aes(x = beta, fill = grupo, color = grupo)) +
  geom_density(alpha = 0.35, linewidth = 0.8) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "gray30") +
  scale_fill_manual(values  = c(col_ok, col_mc)) +
  scale_color_manual(values = c(col_ok, col_mc)) +
  labs(
    title    = expression(paste("Distribución del estimador ", hat(beta)[1])),
    subtitle = "Valor verdadero = 1  |  300 muestras simuladas",
    x        = expression(hat(beta)[1]),
    y        = "Densidad",
    fill     = NULL, color = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "gray40", size = 11),
    legend.position  = "bottom",
    panel.grid.minor = element_blank()
  )

# --- Componer ---
grid.arrange(
  arrangeGrob(p1, p2, ncol = 2),
  p3,
  nrow = 2,
  top = grid::textGrob(
    "Multicolinealidad: qué es y qué hace",
    gp = grid::gpar(fontsize = 16, fontface = "bold")
  )
)