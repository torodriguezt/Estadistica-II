library(ggplot2)

set.seed(42)
n   <- 150
x   <- runif(n, 1, 10)
b0  <- 1.0
b1  <- 0.18
lam <- exp(b0 + b1 * x)
y   <- rpois(n, lam)

datos_sim <- data.frame(x = x, y = y)
curva     <- data.frame(
  x   = seq(1, 10, length.out = 300),
  lam = exp(b0 + b1 * seq(1, 10, length.out = 300))
)

ggplot() +
  geom_point(
    data  = datos_sim,
    aes(x = x, y = y),
    color = "#BA7517",
    size  = 2, alpha = 0.55
  ) +
  geom_line(
    data      = curva,
    aes(x = x, y = lam),
    color     = "#0F6E56",
    linewidth = 1.4
  ) +
  geom_hline(
    yintercept = mean(datos_sim$y),
    linetype   = "dashed",
    color      = "gray50",
    linewidth  = 0.7
  ) +
  annotate(
    "text",
    x     = 9.8,
    y     = mean(datos_sim$y) + 0.4,
    label = paste0("Media global = ", round(mean(datos_sim$y), 2)),
    size  = 3.5, color = "gray45", hjust = 1
  ) +
  scale_y_continuous(breaks = seq(0, ceiling(max(datos_sim$y)), by = 2)) +
  scale_x_continuous(breaks = 1:10) +
  labs(
    title    = "Regresión Poisson — Tasa esperada λ̂",
    subtitle = expression(ln(lambda[i]) == beta[0] + beta[1]*x ~~ "  (datos simulados)"),
    x        = "x  (predictor continuo)",
    y        = expression(hat(lambda)[i] == E(Y[i] ~ "|" ~ x[i]))
  ) +
  theme_bw(base_size = 13) +
  theme(
    plot.title       = element_text(face = "bold"),
    plot.subtitle    = element_text(color = "gray40", size = 11),
    panel.grid.minor = element_blank()
  )