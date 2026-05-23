# Instalar paquetes si no los tienes:
# install.packages(c("ggplot2", "dplyr"))

library(ggplot2)
library(dplyr)

# 1. Simulación de un nuevo conjunto de datos (Más denso y con otros parámetros)
set.seed(854)
n <- 150
x <- runif(n, 10, 50)
# Nueva relación lineal con mayor densidad de puntos
y <- 25 + 1.8 * x + rnorm(n, mean = 0, sd = 8)

datos <- data.frame(x, y)

# 2. Ajuste del modelo de regresión lineal simple
modelo <- lm(y ~ x, data = datos)

# 3. Crear un nuevo data frame con una secuencia fina de X para trazar las bandas suaves
x_seq <- seq(min(x) - 5, max(x) + 5, length.out = 300)
nuevos_datos <- data.frame(x = x_seq)

# 4. Calcular Intervalos de Confianza (para la media E[Y|X])
ic_confianza <- predict(modelo, newdata = nuevos_datos, interval = "confidence", level = 0.95)

# 5. Calcular Intervalos de Predicción (para un nuevo valor individual Y_0)
ic_prediccion <- predict(modelo, newdata = nuevos_datos, interval = "prediction", level = 0.95)

# 6. Consolidar todo en un único Data Frame para ggplot2
df_bandas <- data.frame(
  x = x_seq,
  ajuste = ic_confianza[, "fit"],
  
  # Límites de Confianza (Media)
  conf_lwr = ic_confianza[, "lwr"],
  conf_upr = ic_confianza[, "upr"],
  
  # Límites de Predicción (Individual)
  pred_lwr = ic_prediccion[, "lwr"],
  pred_upr = ic_prediccion[, "upr"]
)

# 7. Gráfica comparativa con ggplot2 (Sin subtítulo)
ggplot() +
  # Capa 1: Banda del Intervalo de Predicción (Más ancha, capa de fondo)
  geom_ribbon(data = df_bandas, aes(x = x, ymin = pred_lwr, ymax = pred_upr, fill = "Predicción (Individual)"), alpha = 0.2) +
  
  # Capa 2: Banda del Intervalo de Confianza (Más estrecha, capa intermedia)
  geom_ribbon(data = df_bandas, aes(x = x, ymin = conf_lwr, ymax = conf_upr, fill = "Confianza (Media)"), alpha = 0.4) +
  
  # Capa 3: Puntos de los datos reales observados (Más densos)
  geom_point(data = datos, aes(x = x, y = y), color = "grey25", size = 1.8, alpha = 0.6) +
  
  # Capa 4: Recta de regresión estimada
  geom_line(data = df_bandas, aes(x = x, y = ajuste, color = "Recta Ajustada"), linewidth = 1.2) +
  
  # Estética, colores y leyendas
  scale_fill_manual(
    name = "Intervalos al 95%", 
    values = c("Predicción (Individual)" = "#ff7f0e", "Confianza (Media)" = "#1f77b4")
  ) +
  scale_color_manual(name = "", values = c("Recta Ajustada" = "darkred")) +
  theme_bw(base_size = 14) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    legend.box = "vertical",
    legend.margin = margin()
  ) +
  labs(
    title = "Intervalo de Confianza vs. Intervalo de Predicción",
    x = "Variable Predictora (X)",
    y = "Variable Respuesta (Y)"
  )