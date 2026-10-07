if (!require(tidymodels)) {
  install.packages("tidymodels")
  library(tidymodels)
}

crear_particiones <- function(datos, semilla, v = 10) {
  set.seed(semilla)
  data_folds1 <- mc_cv(datos, prop = 0.80, times = 1)
  loo <- rsample::loo_cv(datos)
  data_folds2 <- rsample::manual_rset(loo$splits, loo$id)
  data_folds3 <- vfold_cv(datos, v = v)
  list(validacion = data_folds1, loocv = data_folds2, kfold = data_folds3)
}

validacion_cruzada <- function(formulas, particiones) {
  lm_spec <- linear_reg() %>% set_engine("lm")
  lm_models <- workflow_set(preproc = formulas, models = list(lm = lm_spec))
  cv_results <- lm_models %>% workflow_map(
    fn = "fit_resamples",
    resamples = particiones,
    metrics = metric_set(rmse),
    control = control_resamples(save_pred = TRUE)
  )
  respuesta <- all.vars(formulas[[1]])[1]
  collect_predictions(cv_results, summarize = FALSE) %>%
    group_by(wflow_id) %>%
    rmse(truth = !!sym(respuesta), estimate = .pred) %>%
    transmute(modelo = sub("_lm$", "", wflow_id),
              RMSE = .estimate, MSE = .estimate^2) %>%
    arrange(MSE)
}
