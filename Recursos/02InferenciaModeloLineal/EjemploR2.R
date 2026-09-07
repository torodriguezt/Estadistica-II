par(mfrow = c(1,3), mar = c(4.5, 4.5, 3.5, 1.5), family = "sans",
    bg = "white", fg = "grey30")

palette_pts  <- adjustcolor("#2171B5", alpha.f = 0.55)
palette_line <- "#E31A1C"
palette_true <- "#238B45"
palette_r2   <- "#333333"

set.seed(123)
n <- 100
x1 <- seq(0, 4, length.out = n)
y_true1 <- exp(0.8 * x1)
y1 <- y_true1 + rnorm(n, 0, 2)
mod1 <- lm(y1 ~ x1)
R2_1 <- round(summary(mod1)$r.squared, 3)

plot(x1, y1,
     pch = 16, cex = 0.9,
     col = palette_pts,
     main = expression(bold("R"^2 ~ "alto — Modelo mal especificado")),
     xlab = "X", ylab = "Y",
     cex.lab = 1.1, cex.main = 1.05,
     col.axis = "grey40", col.lab = "grey30",
     bty = "l", las = 1)
grid(col = "grey90", lty = 1)
points(x1, y1, pch = 16, cex = 0.9, col = palette_pts)
abline(mod1, col = palette_line, lwd = 2.5)
lines(x1, y_true1, col = palette_true, lwd = 2.5, lty = 2)
legend("topleft",
       legend = c("Datos", "Recta lineal", "Relación verdadera"),
       col = c(palette_pts, palette_line, palette_true),
       lwd = c(NA, 2.5, 2.5), lty = c(NA, 1, 2),
       pch = c(16, NA, NA), pt.cex = 1,
       cex = 0.75, bty = "n", y.intersp = 1.2)
legend("bottomright",
       legend = bquote(R^2 == .(R2_1)),
       bty = "n", cex = 1.05, text.col = palette_r2, text.font = 2)



x2 <- runif(n, -3, 3)
y2 <- x2^2 + rnorm(n, 0, 1)
mod2 <- lm(y2 ~ x2)
R2_2 <- round(summary(mod2)$r.squared, 3)

plot(x2, y2,
     pch = 16, cex = 0.9,
     col = palette_pts,
     main = expression(bold("R"^2 %~~% "0 — Relación no lineal")),
     xlab = "X", ylab = "Y",
     cex.lab = 1.1, cex.main = 1.05,
     col.axis = "grey40", col.lab = "grey30",
     bty = "l", las = 1)
grid(col = "grey90", lty = 1)
points(x2, y2, pch = 16, cex = 0.9, col = palette_pts)
abline(mod2, col = palette_line, lwd = 2.5)
x_ord <- sort(x2)
lines(x_ord, x_ord^2, col = palette_true, lwd = 2.5, lty = 2)
legend("top",
       legend = c("Datos", "Recta lineal", "Relación verdadera"),
       col = c(palette_pts, palette_line, palette_true),
       lwd = c(NA, 2.5, 2.5), lty = c(NA, 1, 2),
       pch = c(16, NA, NA), pt.cex = 1,
       cex = 0.75, bty = "n", y.intersp = 1.2)
legend("topright",
       legend = bquote(R^2 == .(R2_2)),
       bty = "n", cex = 1.05, text.col = palette_r2, text.font = 2)



x3 <- runif(n, 0, 10)
y3 <- 3 + 2*x3 + rnorm(n, 0, 2)
mod3 <- lm(y3 ~ x3)
R2_3 <- round(summary(mod3)$r.squared, 3)

plot(x3, y3,
     pch = 16, cex = 0.9,
     col = palette_pts,
     main = expression(bold("Modelo correcto — Buen ajuste")),
     xlab = "X", ylab = "Y",
     cex.lab = 1.1, cex.main = 1.05,
     col.axis = "grey40", col.lab = "grey30",
     bty = "l", las = 1)
grid(col = "grey90", lty = 1)
points(x3, y3, pch = 16, cex = 0.9, col = palette_pts)
abline(mod3, col = palette_line, lwd = 2.5)
legend("topleft",
       legend = c("Datos", "Recta ajustada"),
       col = c(palette_pts, palette_line),
       lwd = c(NA, 2.5), lty = c(NA, 1),
       pch = c(16, NA), pt.cex = 1,
       cex = 0.75, bty = "n", y.intersp = 1.2)
legend("bottomright",
       legend = bquote(R^2 == .(R2_3)),
       bty = "n", cex = 1.05, text.col = palette_r2, text.font = 2)

par(mfrow = c(1,1))