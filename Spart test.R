data <- as.data.frame(UCBAdmissions)

sc <- spark_connect(master = "local") 
import_UCBA <- copy_to(sc, data, "spark_UCBA",
                       overwrite = TRUE) 
# fit a Bernoulli naive Bayes model
model <- ml_naive_bayes(import_UCBA, Admit ~ Gender + Dept, smoothing = 0)
 
 # get the summary of the model
 summary(model)
 
 # make predictions
 predictions <- predict(model, import_UCBA)
