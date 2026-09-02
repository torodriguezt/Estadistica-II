cedulas <- c(1000111222, 1000333444, 1000555666)   # <-- EDITA AQUI. INSERTE LAS CÉDULAS

if (!requireNamespace("MASS", quietly = TRUE)) install.packages("MASS")
library(MASS)

.z <- as.integer(sum(as.numeric(cedulas)) %% 100000)
n <- 500

.q1 <- c(391092600L, 390723000L, 278593360L, 391092593L)
.q2 <- c(52131L, 51829L, 51855L, 51840L, 51855L, 51737L, 52131L)
.q3 <- c(51857L, 13694L, 51855L, 13681L, 51855L, 51857L, 51925L)
.t1 <- c(76L, 122L, 123L, 122L, 113L, 107L, 126L, 109L, 118L, 112L, 99L,
         82L, 118L, 103L, 107L, 112L, 99L, 82L, 118L, 120L, 109L, 126L,
         107L, 112L, 109L, 118L, 112L)
.t2 <- c(71L, 46L, 99L, 71L, 45L, 99L, 71L, 44L, 99L, 71L, 43L, 99L, 72L,
         99L, 92L, 112L, 113L, 108L, 106L, 114L, 112L, 99L, 90L, 113L,
         121L, 122L, 109L, 114L, 112L)

.f <- function(x, m) (bitwXor(x, m) - 32768L) / 100
.s <- function(x) strsplit(intToUtf8(bitwXor(x, 31L)), "|", fixed = TRUE)[[1]]

.k <- ((.z * 7919 + 12345) %% 101) %% 4 + 1
.v <- bitwXor(.q1[.k], 305419896L)
.a <- c(.v %/% 1e6, (.v %/% 1e4) %% 100, (.v %/% 100) %% 100, .v %% 100) /
  c(100, 10, 10, 10)
.w0 <- .s(.t1)
.nm <- .s(.t2)

set.seed(.z)

.b <- .f(.q2, 19087L) * runif(7, 0.85, 1.15)
.g <- .f(.q3, 19087L) * runif(7, 0.85, 1.15)

.r <- matrix(c(
  1.00, 0.10, 0.30, 0.25,
  0.10, 1.00, 0.05, 0.05,
  0.30, 0.05, 1.00, .a[1],
  0.25, 0.05, .a[1], 1.00
), nrow = 4, byrow = TRUE)
.mu <- c(5, 40, 60, 55)
.sg <- c(2, 10, 15, 20)

.xc <- mvrnorm(n, mu = .mu, Sigma = diag(.sg) %*% .r %*% diag(.sg))
.x1 <- pmax(.xc[, 1], 0.5)
.x2 <- round(pmin(pmax(.xc[, 2], 18), 80))
.x3 <- pmin(pmax(.xc[, 3], 0), 100)
.x4 <- pmin(pmax(.xc[, 4], 0), 100)
.wl <- sample(.w0, n, TRUE, c(0.4, 0.4, 0.2))
.d1 <- as.integer(.wl == .w0[2])
.d2 <- as.integer(.wl == .w0[3])

.u1 <- rnorm(n)
.u2 <- rnorm(n)
.m <- .b[1] + .b[2] * .x1 + .b[3] * .x2 + .b[4] * .x3 + .b[5] * .x4 +
  .b[6] * .d1 + .b[7] * .d2 + .a[4] * ((.x3 - .mu[3]) / 10)^2
.e <- 3 * ((1 - .a[3]) * .u1 + .a[3] * (.u2^2 - 1) / sqrt(2)) *
  (.x1 / mean(.x1))^.a[2]
.y1 <- round(.m + .e, 2)

.h <- .g[1] + .g[2] * .x1 + .g[3] * .x2 + .g[4] * .x3 + .g[5] * .x4 +
  .g[6] * .d1 + .g[7] * .d2
.y2 <- rbinom(n, 1, 1 / (1 + exp(-.h)))

datos <- data.frame(round(.x1, 2), .x2, round(.x3, 1), round(.x4, 1),
                    .wl, .y1, .y2)
names(datos) <- .nm
write.csv(datos, "datos.csv", row.names = FALSE)

cat("=====================================================\n")
cat(" Base generada correctamente (datos / datos.csv)\n")
cat(" TU SEMILLA ES:", .z, "\n")
cat(" Para la particion train/test usa: set.seed(", .z, ")\n", sep = "")
cat(" CODIGO DE VERIFICACION (reportalo en la portada):",
    bitwXor(as.integer(.z * 8L + (.k - 1L)), 2731L), "\n")
cat("=====================================================\n")

set.seed(.z)
idx <- sample(1:nrow(datos), size = 0.7 * nrow(datos))
data_train <- datos[idx, ]
data_test  <- datos[-idx, ]
