library(dplyr)
library(ggplot2)
library(data.table)
library(recosystem)

products             <- fread('Data/products.csv')
aisles               <- fread('Data/aisles.csv')
departments          <- fread('Data/departments.csv')

orders               <- fread("Data/orders.csv")
orderTrain           <- fread('Data/order_products__train.csv')
orderPrior           <- fread('Data/order_products__prior.csv')

# собираем все заказы, где у нас извесен списко товара
orderAllProducts <- rbind(orderTrain, orderPrior)

# рейтинг товара - количество его покупок конретным покупателем
# делаем таблицу - покупатель, товар и кол-во его покупок
orders %>%
  filter(eval_set != "test") %>%
  left_join(orderAllProducts, by = "order_id") %>%
  group_by(user_id, product_id) %>%
  count(product_id) -> userRating

startTimeCalculation <- proc.time()
opts = r$tune(
  trainSet,
  opts = list(
    dim = c(10, 20, 30), # dim Integer, the number of latent factors. Default is 10.
    # lrate - Numeric, the learning rate, which can be thought of as the step size 
    # in gradient descent.
    # Default is 0.1.
    lrate = c(0.1, 0.2), 
    # nmf Logical, whether to perform non-negative matrix factorization. Default is FALSE.
    nmf = TRUE,
    # The loss option may take the following values:
    #   For real-valued matrix factorization,
    # "l2" Squared error (L2-norm)
    # "l1" Absolute error (L1-norm)
    # "kl" Generalized KL-divergence
    # For binary matrix factorization,
    # "log" Logarithmic error
    # "squared_hinge" Squared hinge loss
    # "hinge" Hinge loss
    # For one-class matrix factorization,
    # "row_log" Row-oriented pair-wise logarithmic loss
    # "col_log" Column-oriented pair-wise logarithmic loss
    loss = "l2",
    # nthread Integer, the number of threads for parallel computing. Default is 1.
    nthread = 4,
    niter = 20 # niter Integer, the number of iterations. Default is 20.
  )
)
startTimeCalculation - proc.time()

opts
# Тренировочный набор данных с рейтингами
trainSet <-
  data_memory(userRating$user_id, userRating$product_id, rating = userRating$n)

r = Reco()  # инициализация

# Тренируем с параметрами по умолчанию
startTimeCalculation <- proc.time()
r$train(trainSet) 
startTimeCalculation - proc.time()
# user  system elapsed 
# -26.252  -0.579 -26.857 

# iter      tr_rmse          obj
# 0       3.2347   1.5385e+08
# 1       3.1000   1.4261e+08
# 2       3.0553   1.3925e+08
# 3       3.0249   1.3719e+08
# 4       2.9993   1.3548e+08
# 5       2.9770   1.3402e+08
# 6       2.9575   1.3276e+08
# 7       2.9402   1.3168e+08
# 8       2.9241   1.3070e+08
# 9       2.9089   1.2975e+08
# 10       2.8950   1.2893e+08
# 11       2.8815   1.2813e+08
# 12       2.8685   1.2737e+08
# 13       2.8564   1.2667e+08
# 14       2.8447   1.2600e+08
# 15       2.8335   1.2535e+08
# 16       2.8230   1.2475e+08
# 17       2.8128   1.2416e+08
# 18       2.8035   1.2366e+08
# 19       2.7943   1.2314e+08

testSet <- data_memory(userRating$user_id[1:10000], userRating$product_id[1:10000])
predictionSet <- r$predict(testSet, out_memory())

sqrt(sum((predictionSet-userRating$n[1:10000])^2))/10000
# [1] 0.0244997

## Write P and Q matrices to files
P_file = out_file(tempfile())
Q_file = out_file(tempfile())
r$output(P_file, Q_file)
head(read.table(P_file@dest, header = FALSE, sep = " "))
head(read.table(Q_file@dest, header = FALSE, sep = " "))
