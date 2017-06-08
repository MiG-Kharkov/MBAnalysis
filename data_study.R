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

# Тренировочный набор данных с рейтингами
trainSet <-
  data_memory(userRating$user_id, userRating$product_id, rating = userRating$n, index1 = TRUE)
r = Reco()  # инициализация

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
# user     system    elapsed 
# -26344.240    -79.609  -6713.053 
opts$res



# Тренируем с параметрами по умолчанию
startTimeCalculation <- proc.time()
r$train(trainSet, opts = c(opts$min, nthread = 4, niter = 20,nmf = TRUE)) 
startTimeCalculation - proc.time()
# user  system elapsed 
# -54.275  -1.047 -12.573 


testSet <- data_memory(userRating$user_id[1:10000], userRating$product_id[1:10000])
predictionSet <- r$predict(testSet, out_memory())

sqrt(sum((predictionSet-userRating$n[1:10000])^2))/10000
# [1] 0.0244997
# after tuning 0.02482675
# [1] 0.02505054 if nmf = TRUE - non negative 

## Write P and Q matrices to files
P_file = out_file("p_out.txt")
Q_file = out_file("q_out.txt")
r$output(P_file, Q_file)
Pmatrix <- read.table(P_file@dest, header = FALSE, sep = " ")
Qmatrix <- read.table(Q_file@dest, header = FALSE, sep = " ")

