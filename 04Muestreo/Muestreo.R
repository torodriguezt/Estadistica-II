############################################################
#       Función para Estimación por Muestreo
#      Detecta automáticamente MAS y MAE
#   - MAS: datos_muestrales = vector,  info_poblacion = N
#   - MAE: datos_muestrales = lista,   info_poblacion = c(N_1, N_2, ...)
############################################################
estimar_parametro <- function(datos_muestrales,
                              info_poblacion,
                              parametro,
                              intervalo = FALSE,
                              confianza = 0.95){

  # =================================================================
  #   CASO 1: MUESTREO ALEATORIO ESTRATIFICADO (MAE) -> lista
  # =================================================================
  if (is.list(datos_muestrales)) {
    muestras_por_estrato <- datos_muestrales
    N_por_estrato <- info_poblacion

    if (!is.vector(N_por_estrato) || length(muestras_por_estrato) != length(N_por_estrato)) {
      stop("Para MAE, 'info_poblacion' debe ser un vector con tamaños por estrato")
    }

    L <- length(muestras_por_estrato)
    N <- sum(N_por_estrato)
    pesos_W_h <- N_por_estrato / N

    n_h <- sapply(muestras_por_estrato, length)
    media_h <- sapply(muestras_por_estrato, mean)
    var_muestral_h <- sapply(muestras_por_estrato, var)
    fpc_h <- (N_por_estrato - n_h) / N_por_estrato

    if (parametro == "mu") {
      estimador <- sum(pesos_W_h * media_h)
      var_estimador <- sum((pesos_W_h^2 * var_muestral_h / n_h) * fpc_h)
    } else if (parametro == "tau") {
      estimador <- N * sum(pesos_W_h * media_h)
      var_estimador <- sum((N_por_estrato^2 * var_muestral_h / n_h) * fpc_h)
    } else if (parametro == "p") {
      p_h <- media_h
      estimador <- sum(pesos_W_h * p_h)
      var_p_h <- (p_h * (1 - p_h)) / (n_h - 1)
      var_estimador <- sum((pesos_W_h^2 * var_p_h) * fpc_h)
    } else if (parametro == "A") {
      p_h <- media_h
      estimador <- N * sum(pesos_W_h * p_h)
      var_p_h <- (p_h * (1 - p_h)) / (n_h - 1)
      var_estimador <- N^2 * sum((pesos_W_h^2 * var_p_h) * fpc_h)
    } else {
      stop("Parámetro para MAE no reconocido. Usar 'mu', 'tau', 'p' o 'A'")
    }

    if (intervalo) {
      error_est <- sqrt(var_estimador)
      grados_libertad <- sum(n_h) - L
      t_critico <- qt(1 - (1 - confianza) / 2, df = grados_libertad)
      ic <- c(estimador - t_critico * error_est, estimador + t_critico * error_est)
    }

  # =================================================================
  #   CASO 2: MUESTREO ALEATORIO SIMPLE (MAS) -> vector
  # =================================================================
  } else if (is.vector(datos_muestrales)) {
    muestra <- datos_muestrales
    N <- info_poblacion
    if (!is.numeric(N) || length(N) != 1) {
      stop("Para MAS, 'info_poblacion' debe ser un único número (N)")
    }

    n <- length(muestra)
    media_muestra <- mean(muestra)
    sd_muestra <- sd(muestra)
    fpc <- sqrt((N - n) / N)

    if (parametro == "mu") {
      estimador <- media_muestra
      var_estimador <- ((sd_muestra / sqrt(n)) * fpc)^2
    } else if (parametro == "tau") {
      estimador <- media_muestra * N
      var_estimador <- ((sd_muestra / sqrt(n)) * (N * fpc))^2
    } else if (parametro == "p") {
      p_muestra <- mean(muestra)
      estimador <- p_muestra
      var_estimador <- (sqrt((p_muestra * (1 - p_muestra)) / (n - 1)) * fpc)^2
    } else if (parametro == "A") {
      p_muestra <- mean(muestra)
      estimador <- N * p_muestra
      var_estimador <- (N * sqrt((p_muestra * (1 - p_muestra)) / (n - 1)) * fpc)^2
    } else {
      stop("Parámetro para MAS no reconocido. Usar 'mu', 'tau', 'p' o 'A'")
    }

    if (intervalo) {
      error_est <- sqrt(var_estimador)
      t_critico <- qt(1 - (1 - confianza) / 2, df = n - 1)
      ic <- c(estimador - t_critico * error_est, estimador + t_critico * error_est)
    }

  } else {
    stop("El formato de 'datos_muestrales' no es válido (vector para MAS, lista para MAE)")
  }

  resultado <- list(Estimador = estimador, Varianza = var_estimador)
  if (exists("ic") && !is.null(ic)) resultado$IC <- ic
  return(resultado)
}
