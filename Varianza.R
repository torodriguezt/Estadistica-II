# Instalar paquetes si no los tienes:
# install.packages(c("ggplot2", "dplyr"))

library(ggplot2)
library(dplyr)

# 1. Configuración de la simulación
set.seed(2026)
n <- 400
x <- runif(n, 10, 50)

# ---------------------------------------------------------
# ESCENARIO 1: Ideal (Homocedasticidad)
# El error tiene varianza constante en todo el dominio de X
y1 <- 10 + 3 * x + rnorm(n, mean = 0, sd = 15)
mod1 <- lm(y1 ~ x)

# ---------------------------------------------------------
# ESCENARIO 2: Embudo / Megáfono (Heterocedasticidad)
# La varianza crece a medida que crece X (ej. sd = 0.8 * x)
y2 <- 10 + 3 * x + rnorm(n, mean = 0, sd = 0.8 * x)
mod2 <- lm(y2 ~ x)

# ---------------------------------------------------------
# ESCENARIO 3: Balón de fútbol / Diamante (Heterocedasticidad)
# Varianza baja en los extremos, alta en el medio. 
# Usamos una función seno para simular el abultamiento central.
sd_balon <- 2 + 15 * sin(pi * (x - 10) / 40) 
y3 <- 10 + 3 * x + rnorm(n, mean = 0, sd = sd_balon)
mod3 <- lm(y3 ~ x)

# ---------------------------------------------------------
# ESCENARIO 4: Tendencia No Lineal (Mala especificación)
# El modelo real tiene un término cuadrático, pero ajustamos un modelo lineal simple
y4 <- -50 + 15 * x - 0.2 * x^2 + rnorm(n, mean = 0, sd = 10)
mod4 <- lm(y4 ~ x) # ¡Ajustamos un modelo lineal equivocado a propósito!

# ---------------------------------------------------------
# 2. Consolidar los datos para graficar
df_patrones <- data.frame(
  Ajustados = c(fitted(mod1), fitted(mod2), fitted(mod3), fitted(mod4)),
  Residuales = c(resid(mod1), resid(mod2), resid(mod3), resid(mod4)),
  Patron = factor(rep(c(
    "1. Ideal - Homocedástico",
    "2. Embudo (Varianza Creciente)",
    "3. Balón de Fútbol (Varianza no monótona)",
    "4. Curva en U (Falta término cuadrático)"
  ), each = n))
)

# 3. Creación del gráfico con ggplot2
ggplot(df_patrones, aes(x = Ajustados, y = Residuales, color = Patron)) +
  geom_point(alpha = 0.5, size = 1.5) +
  
  # Línea del cero de referencia
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.8) +
  
  # Línea de tendencia suave (Loess) para ver el comportamiento promedio del residual
  geom_smooth(method = "loess", se = FALSE, color = "black", linewidth = 1.2) +
  
  # Facetamiento en 4 paneles
  facet_wrap(~Patron, scales = "free") +
  
  # Colores específicos para cada patrón
  scale_color_manual(values = c(
    "1. Ideal - Homocedástico" = "#2ca02c", # Verde = Bien
    "2. Embudo (Varianza Creciente)" = "#d62728",              # Rojo = Alerta
    "3. Balón de Fútbol (Varianza no monótona)" = "#ff7f0e",   # Naranja = Alerta
    "4. Curva en U (Falta término cuadrático)" = "#1f77b4"     # Azul = Estructural
  )) +
  
  # Estética y etiquetas
  theme_bw(base_size = 13) +
  theme(
    legend.position = "none", # Ocultamos leyenda porque los títulos ya lo dicen
    strip.text = element_text(face = "bold", size = 11, color = "white"),
    strip.background = element_rect(fill = "#333333"),
    plot.title = element_text(face = "bold", size = 16)
  ) +
  labs(
    title = "Patrones de Diagnóstico en Gráficos de Residuales",
    subtitle = "Análisis de Residuales Ordinarios vs. Valores Ajustados",
    x = "Valores Ajustados",
    y = "Residuales"
  )