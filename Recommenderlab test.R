# install.packages("recommenderlab")
library(recommenderlab)

set.seed(123)
m <- matrix(
  sample(c(as.numeric(1:5), NA), 50, replace = TRUE,prob = c(rep(.4 / 6, 5), .6)),
  ncol = 10,
  dimnames = list(
    user = paste("u", 1:5, sep = ""),
    item = paste("i", 1:10, sep = "")
  )
)
# Преобразовать в рейтинговую матрицу
r <- as(m, "realRatingMatrix")

getRatingMatrix(r)
# Преобразовать назад и проверить 
identical(as(r, "matrix"),m)
as(r, "list")
as(r, "data.frame")

# Нормализация 
r_m <- normalize(r)
getRatingMatrix(r_m)

getRatingMatrix(denormalize(r_m))
image(r, main = "Raw Ratings")
image(r_m, main = "Normalized Ratings")


r_b <- binarize(r, minRating=3)
as(r_b, "matrix")


recommenderRegistry$get_entries(dataType = "realRatingMatrix")

data(Jester5k)

rec <- Recommender(r[1:4], method = "POPULAR")
names(getModel(rec))

recom <- predict(rec, r[5], n=5)
recom
as(recom, "matrix")
recom2 <- bestN(recom, n = 2)
recom2
as(recom2, "matrix")


recom <- predict(rec, r[5], type="ratings")
as(recom, "matrix")

recom <- predict(rec, r[5], type="ratingMatrix")
as(recom, "matrix")

library(recosystem) 
recoObj = Reco()  # инициализация
userRating <- as(r, "list")

trainSet <-
  data_memory(userRating$user_id, userRating$product_id, rating = userRating$averRate, index1 = TRUE)