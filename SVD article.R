# https://www.r-bloggers.com/matrix-factorization/

unroll_Vecs <- function (params, Y, R, num_users, num_movies, num_features) {
  # Unrolls vector into X and Theta
  # Also calculates difference between preduction and actual 
  
  endIdx <- num_movies * num_features
  
  X     <- matrix(params[1:endIdx], nrow = num_movies, ncol = num_features)
  Theta <- matrix(params[(endIdx + 1): (endIdx + (num_users * num_features))], 
                  nrow = num_users, ncol = num_features)
  
  Y_dash     <-   (((X %*% t(Theta)) - Y) * R) # Prediction error
  
  return(list(X = X, Theta = Theta, Y_dash = Y_dash))
}

J_cost <-  function(params, Y, R, num_users, num_movies, num_features, lambda, alpha) {
  # Calculates the cost
  
  unrolled <- unroll_Vecs(params, Y, R, num_users, num_movies, num_features)
  X <- unrolled$X
  Theta <- unrolled$Theta
  Y_dash <- unrolled$Y_dash
  
  J <-  .5 * sum(   Y_dash ^2)  + lambda/2 * sum(Theta^2) + lambda/2 * sum(X^2)
  
  return (J)
}

grr <- function(params, Y, R, num_users, num_movies, num_features, lambda, alpha) {
  # Calculates the gradient step
  # Here lambda is the regularization parameter
  # Alpha is the step size
  
  unrolled <- unroll_Vecs(params, Y, R, num_users, num_movies, num_features)
  X <- unrolled$X
  Theta <- unrolled$Theta
  Y_dash <- unrolled$Y_dash
  
  X_grad     <- ((   Y_dash  %*% Theta) + lambda * X     )
  Theta_grad <- (( t(Y_dash) %*% X)     + lambda * Theta )
  
  grad = c(X_grad, Theta_grad)
  return(grad)
}

# Now that everything is set up, call optim
num_users <- 1000
num_features <- 100
num_movies <- 10
maxit <- 10
print(
  res <- optim(par = c(runif(num_users * num_features), runif(num_movies * num_features)), # Random starting parameters
               fn = J_cost, gr = grr, 
               Y=Y, R=R, 
               num_users=num_users, num_movies=num_movies,num_features=num_features, 
               lambda=lambda, alpha = alpha, 
               method = "L-BFGS-B", control=list(maxit=maxit, trace=1))
)


require(recommenderlab) # Install this if you don't have it already
require(devtools) # Install this if you don't have this already
# Get additional recommendation algorithms
install_github("sanealytics", "recommenderlabrats")

data(MovieLense) # Get data

# Divvy it up
scheme <- evaluationScheme(MovieLense, method = "split", train = .9,
                           k = 1, given = 10, goodRating = 4) 

scheme

# register recommender
recommenderRegistry$set_entry(
  method="RSVD", dataType = "realRatingMatrix", fun=REAL_RSVD,
  description="Recommender based on Low Rank Matrix Factorization (real data).")

# Some algorithms to test against
algorithms <- list(
  "random items" = list(name="RANDOM", param=list(normalize = "Z-score")),
  "popular items" = list(name="POPULAR", param=list(normalize = "Z-score")),
  "user-based CF" = list(name="UBCF", param=list(normalize = "Z-score",
                                                 method="Cosine",
                                                 nn=50, minRating=3)),
  "Matrix Factorization" = list(name="RSVD", param=list(categories = 10, 
                                                        lambda = 10,
                                                        maxit = 100))
)

# run algorithms, predict next n movies
results <- evaluate(scheme, algorithms, n=c(1, 3, 5, 10, 15, 20))

# Draw ROC curve
plot(results, annotate = 1:4, legend="topleft")

# See precision / recall
plot(results, "prec/rec", annotate=3)