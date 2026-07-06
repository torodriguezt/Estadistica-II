library(ggplot2)

set.seed(42)
n   <- 200
x   <- seq(-6, 6, length.out = n)
b0  <- 0
b1  <- 1.5
p   <- 1 / (1 + exp(-(b0 + b1 * x)))
y   <- rbinom(n, 1, p)

datos_sim <- data.frame(x = x, y = y)
curva     <- data.frame(x = x, p = p)

ggplot() +
  geom_jitter(
    data   = datos_sim,
    aes(x = x, y = y, color = factor(y)),
    width  = 0.08, height = 0.02,
    size   = 1.8, alpha = 0.55
  ) +
  geom_line(
    data      = curva,
    aes(x = x, y = p),
    color     = "#2B7BB9",
    linewidth = 1.4
  ) +
  geom_hline(
    yintercept = 0.5,
    linetype   = "dashed",
    color      = "gray50",
    linewidth  = 0.7
  ) +
  annotate(
    "text", x = 5.8, y = 0.53,
    label = "θ* = 0.5", size = 3.5, color = "gray45", hjust = 1
  ) +
  scale_color_manual(
    values = c("0" = "#888780", "1" = "#D85A30"),
    labels = c("0" = "y = 0  (fracaso)", "1" = "y = 1  (éxito)"),
    name   = NULL
  ) +
  scale_y_continuous(
    breaks = c(0, 0.25, 0.5, 0.75, 1),
    limits = c(-0.06, 1.06)
  ) +
  labs(
    title    = "Regresión Logística — Curva sigmoide",
    subtitle = expression(logit(theta[i]) == beta[0] + beta[1]*x ~~ "  (datos simulados)"),
    x        = "x  (predictor continuo)",
    y        = expression(hat(theta)[i] == P(Y[i] == 1 ~ "|" ~ x[i]))
  ) +
  theme_bw(base_size = 13) +
  theme(
    plot.title       = element_text(face = "bold"),
    plot.subtitle    = element_text(color = "gray40", size = 11),
    legend.position  = "top",
    panel.grid.minor = element_blank()
  )