library(pROC)
library(PRROC)

set.seed(42)

data <- read.csv("data/nacimientos_simplificado.csv")

n <- nrow(data)

train_idx <- sample(1:n, size = 0.7 * n)

train <- data[train_idx, ]
test  <- data[-train_idx, ]

#train <- datos[1:1400]
#test <- datos[1401:2000,]


modelo <- glm(BAJO_PESO ~ ., data = train, family = binomial(link = "logit"))

summary(modelo)

predict_test <- predict(modelo, newdata = test, type = "response")
clase <- ifelse(predict_test >= 0.5, 1, 0)

g <- data.frame(real = test$BAJO_PESO, prob = predict_test, pred = clase)


roc <- roc.curve(scores.class0 = predict_test, weights.class0 = test$BAJO_PESO, curve = TRUE)
plot(roc)


roc_cut <- roc(test$BAJO_PESO, predict_test)
coords(roc_cut, "best", best.method = "closest.topleft")$threshold



matriz1 <- table(test$BAJO_PESO, predict_test > 0.103)
TCC1 <- sum(diag(matriz1)) / sum(matriz1)

matriz2 <- table(test$BAJO_PESO, predict_test > 0.25)
TCC2 <- sum(diag(matriz2))/sum(matriz2)

matriz3 <- table(test$BAJO_PESO, predict_test > 0.5)
TCC3 <- sum(diag(matriz3))/sum(matriz3)

