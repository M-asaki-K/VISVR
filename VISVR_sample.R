#Packages installation
library(readr) 
library(dplyr) 
library(assertr) 
library(rsample)
library(genalg)
library(pls)
library(e1071)
library(kernlab)
library(iml)
library(devtools)

pkgs <- c('foreach', 'doParallel')
lapply(pkgs, require, character.only = T)
#if you want to change the number of threads for the calculation, please change the value "detectCores()"
registerDoParallel(makeCluster(detectCores()))

#データベースの読み込み
path <- file.choose()
path

compounds <- read.csv(path)
View(compounds)

#-----------remove some columns if needed（今回は1, 3, 4列目を除去）--------------
trimed.compounds <- (compounds[, -c(1, 3, 4)])

#-----------select rows without empty cells---------
is.completes.t <- complete.cases(t(trimed.compounds))
is.completes.t

complete.compounds <- trimed.compounds[,is.completes.t]

is.completes <- complete.cases(complete.compounds)
is.completes

complete.compounds <- complete.compounds[is.completes,]
View(complete.compounds)

#-----------select x from the dataset-----------------
x <- as.matrix(complete.compounds[,c(2:802)])
View(x)

#-----------calculate standard distribution of x------
x.sds <- apply(x, 2, sd)
x.sds
#-----------remove columns of 0 distribution from x----
sd.is.not.0 <- x.sds != 0
sd.is.not.0
x <- x[, sd.is.not.0]

#共相関となる列の片方を削除（相関係数の閾値はcutoffで設定、今回は0.95）
library(caret)

df2 <- as.matrix(cor(x))
hc = findCorrelation(df2, cutoff = .95, exact = FALSE) 
hc = sort(hc)
reduced_Data = x[,-c(hc)]

x <- reduced_Data
#reduced_Dataはこの後使わないので、メモリ上から削除
rm(reduced_Data)

y <- complete.compounds[, c(1)]

#--------------------divide into test and training data----------------------
train_size = 0.7

n = nrow(x)
#------------collect the data with n*train_size from the dataset------------
perm = sample(n, size = round(n * train_size))

train_data <- cbind.data.frame(x[perm, ])
test_data <- cbind.data.frame(x[-perm, ])
nrow(test_data)
#-----------select y from the dataset------------------
train_labels <- complete.compounds[, c(1)]
train_labels <- train_labels[perm]
test_labels <- complete.compounds[, c(1)]
test_labels <- test_labels[-perm]
test_labels

# Test data is *not* used when calculating the mean and std.

# Normalize training data
train_data <- scale(train_data) 

is.complete_train <- complete.cases(t(train_data))

# Use means and standard deviations from training set to normalize test set
col_means_train <- attr(train_data, "scaled:center") 
col_stddevs_train <- attr(train_data, "scaled:scale")
test_data <- scale(test_data, center = col_means_train, scale = col_stddevs_train)

#正規化でNaNを吐いた列は削除する
is.complete_test <- complete.cases(t(test_data))
is.complete_total <- is.complete_train&is.complete_test
is.complete_total

train_data <- train_data[ , is.complete_total]
test_data <- test_data[ , is.complete_total]

#----------------randomforest parameter tune-----------------------
library(randomForest)
library(rBayesianOptimization)　#kfoldを行うためだけに登場…もったいない使い方

#グリッドサーチのパラメータ設定
M_num <- c(1, 2, 5, 10, 20)
t_num <- c(100, 200, 500, 1000)

parameters_rf <- expand.grid(M_num = M_num, t_num = t_num)

#交差検証の設定（全てのモデル計算に共通、パッケージに実装されてることもあるのですが、ここも工夫代が色々あるので独立に設定…groupkfoldとか）
nfolds <- 5
folds <- KFold(train_labels, nfolds)
folds

#学習データの設定
Xtrain <- train_data
ytrain <- train_labels

df2 <- as.data.frame(cbind(ytrain,Xtrain))

#各パラメータに対して交差検証による予測精度を測定
result <- foreach(i = 1:nrow(parameters_rf), .combine = rbind, .packages = c("foreach", "doParallel", "readr", "dplyr", "assertr", "rsample", "randomForest")) %dopar% {
  Mn <- parameters_rf[i, ]$M_num
  tn <- parameters_rf[i, ]$t_num

  ### K-FOLD VALIDATION ###
  out <- foreach(j = 1:(nfolds), .combine = rbind,.packages = c("foreach", "doParallel", "readr", "dplyr", "assertr", "rsample", "randomForest"), .inorder = FALSE) %dopar% {
    deve <- df2[folds[[j]], ]
    test <- na.omit(df2[-folds[[j]], ])    
    mdl <- randomForest::randomForest(ytrain~., data = deve, mtry = Mn, ntree = tn)
    pred <- predict(mdl, test)
    data.frame(test[, c(1)], pred)
    
  }
  ### CALCULATE SVM PERFORMANCE ###
  roc <- sum((out[, c(1)] - out[, c(2)])^2) / nrow(out)
  data.frame(parameters_rf[i, ], roc)
}

#最適化されたパラメータを別名で保存
M_num <- min(result[result[, c(3)] <= (min(result[,c(3)])), c(1)])
t_num <- min(result[result[, c(3)] <= (min(result[,c(3)])), c(2)])

#最適化されたランダムフォレストモデルを保存するとともに、重要度を算出
modelVI <- randomForest(ytrain~., data = df2, mtry = M_num, ntree = t_num)
VI <- (modelVI$importance)
VI_std <- as.data.frame(abs((VI - mean(VI)) / sd(VI)))

#重要度でtrain_dataとtest_dataに重みづけするため、データセットの行数に合わせた行列にする
VI_std <- matrix(VI_std, nrow = 1, ncol = ncol(train_data))
VI_matrix_train <- (VI_std[rep(seq_len(nrow(VI_std)), nrow(train_data)), ])
VI_matrix_test <- (VI_std[rep(seq_len(nrow(VI_std)), nrow(test_data)), ])

#重みづけの乗数（候補は3点、金子研究室の論文に掲載されたgithubから引用）
#https://github.com/hkaneko1985/dcekit/blob/master/demo_visvr.py
power_VI <- c(0, 3.1, 0.1)
length(power_VI)

#各重みづけを行ったデータセットに対し、SVMのパラメータを最適化し予測精度を出力
Power_comparison <- foreach(l = 1:length(power_VI), .combine = rbind,.packages = c("foreach", "doParallel", "readr", "dplyr", "assertr", "rsample", "e1071"), .inorder = FALSE)%do%{
  xtrain <- (VI_matrix_train^power_VI[l])*train_data
  ytrain <- train_labels
  df2 <- as.data.frame(cbind(ytrain,xtrain))
  
  ### PARAMETER LIST ###
  cost <- 3
  epsilon <- c(-10,-9,-8,-7,-6,-5,-4,-3,-2,-1, 0)
  
  gam <- foreach(k = -20:10, .combine = rbind,.packages = c("kernlab"))%dopar%{
    k = 1
    rbf <- rbfdot(sigma = 2^k)
    rbf
    
    asmat <- as.matrix(xtrain)
    asmat
    
    kern <- kernelMatrix(rbf, asmat)
    sd(kern)
    data.frame((k +21), sd(kern))
  }
  
  hakata <- which.max(gam$sd.kern.)
  
  gamma <- hakata - 21
  parms <- expand.grid(epsilon = epsilon, cost = cost, gamma = gamma)
  ### LOOP THROUGH PARAMETER VALUES ###
  result <- foreach(i = 1:nrow(parms), .combine = rbind, .packages = c("foreach", "doParallel", "readr", "dplyr", "assertr", "rsample")) %dopar% {
    c <- parms[i, ]$cost
    g <- parms[i, ]$gamma
    e <- parms[i, ]$epsilon
    ### K-FOLD VALIDATION ###
    out <- foreach(j = 1:(nfolds), .combine = rbind,.packages = c("foreach", "doParallel", "readr", "dplyr", "assertr", "rsample"), .inorder = FALSE) %dopar% {
      deve <- df2[folds[[j]], ]
      test <- na.omit(df2[-folds[[j]], ])    
      mdl <- e1071::svm(ytrain~., data = deve, cost = c, gamma = 2^g, epsilon = 2^e)
      pred <- predict(mdl, test)
      data.frame(test[, c(1)], pred)
      
    }
    ### CALCULATE SVM PERFORMANCE ###
    roc <- sum((out[, c(1)] - out[, c(2)])^2) / nrow(out)
    data.frame(parms[i, ], roc)
  }
  
  epsilon <- min(result[result[, c(4)] <= (min(result[,c(4)])), c(1)])
  cost <- c(-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10)
  parms <- expand.grid(epsilon = epsilon, cost = cost, gamma = gamma)
  ### LOOP THROUGH PARAMETER VALUES ###
  result <- foreach(i = 1:nrow(parms), .combine = rbind, .packages = c("foreach", "doParallel", "readr", "dplyr", "assertr", "rsample")) %dopar% {
    c <- parms[i, ]$cost
    g <- parms[i, ]$gamma
    e <- parms[i, ]$epsilon
    ### K-FOLD VALIDATION ###
    out <- foreach(j = 1:(nfolds), .combine = rbind,.packages = c("foreach", "doParallel", "readr", "dplyr", "assertr", "rsample"), .inorder = FALSE) %dopar% {
      deve <- df2[folds[[j]], ]
      test <- na.omit(df2[-folds[[j]], ])    
      mdl <- e1071::svm(ytrain~., data = deve, cost = 2^c, gamma = 2^g, epsilon = 2^e)
      pred <- predict(mdl, test)
      data.frame(test[, c(1)], pred)
      
    }
    ### CALCULATE SVM PERFORMANCE ###
    roc <- sum((out[, c(1)] - out[, c(2)])^2) / nrow(out)
    data.frame(parms[i, ], roc)
  }
  
  cost <- min(result[(result[, c(4)] <= (min(result[,c(4)]))), c(2)])
  gamma <- c(-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10)
  parms <- expand.grid(epsilon = epsilon, cost = cost, gamma = gamma)
  ### LOOP THROUGH PARAMETER VALUES ###
  result <- foreach(i = 1:nrow(parms), .combine = rbind, .packages = c("foreach", "doParallel", "readr", "dplyr", "assertr", "rsample")) %dopar% {
    c <- parms[i, ]$cost
    g <- parms[i, ]$gamma
    e <- parms[i, ]$epsilon
    ### K-FOLD VALIDATION ###
    out <- foreach(j = 1:(nfolds), .combine = rbind,.packages = c("foreach", "doParallel", "readr", "dplyr", "assertr", "rsample"), .inorder = FALSE) %dopar% {
      deve <- df2[folds[[j]], ]
      test <- na.omit(df2[-folds[[j]], ])    
      mdl <- e1071::svm(ytrain~., data = deve, cost = 2^c, gamma = 2^g, epsilon = 2^e)
      pred <- predict(mdl, test)
      data.frame(test[, c(1)], pred)
      
    }
    ### CALCULATE SVM PERFORMANCE ###
    roc <- sum((out[, c(1)] - out[, c(2)])^2) / nrow(out)
    data.frame(parms[i, ], roc)
  }
  
  gamma <- min(result[(result[, c(4)] <= (min(result[,c(4)]))), c(3)])
  bestperformance <- min(result[, c(4)])
  data.frame(power_VI[l], bestperformance)
}

#交差検証の結果が最も優れた重みづけを出力
best_power <- min(Power_comparison[(Power_comparison[, c(2)] <= (min(Power_comparison[,c(2)]))), c(1)])

#最適と判断された重みづけにて、SVMモデルのパラメータを最適化しモデル構築
xtrain <- (VI_matrix_train^best_power)*train_data
xtest <- (VI_matrix_test^best_power)*test_data
ytrain <- train_labels
df2 <- as.data.frame(cbind(ytrain,xtrain))

### PARAMETER LIST ###
cost <- 3
epsilon <- c(-10,-9,-8,-7,-6,-5,-4,-3,-2,-1, 0)

gam <- foreach(k = -20:10, .combine = rbind,.packages = c("kernlab"))%dopar%{
  k = 1
  rbf <- rbfdot(sigma = 2^k)
  rbf
  
  asmat <- as.matrix(xtrain)
  asmat
  
  kern <- kernelMatrix(rbf, asmat)
  sd(kern)
  data.frame((k +21), sd(kern))
}

hakata <- which.max(gam$sd.kern.)

gamma <- hakata - 21
parms <- expand.grid(epsilon = epsilon, cost = cost, gamma = gamma)
### LOOP THROUGH PARAMETER VALUES ###
result <- foreach(i = 1:nrow(parms), .combine = rbind, .packages = c("foreach", "doParallel", "readr", "dplyr", "assertr", "rsample")) %dopar% {
  c <- parms[i, ]$cost
  g <- parms[i, ]$gamma
  e <- parms[i, ]$epsilon
  ### K-FOLD VALIDATION ###
  out <- foreach(j = 1:(nfolds), .combine = rbind,.packages = c("foreach", "doParallel", "readr", "dplyr", "assertr", "rsample"), .inorder = FALSE) %dopar% {
    deve <- df2[folds[[j]], ]
    test <- na.omit(df2[-folds[[j]], ])    
    mdl <- e1071::svm(ytrain~., data = deve, cost = c, gamma = 2^g, epsilon = 2^e)
    pred <- predict(mdl, test)
    data.frame(test[, c(1)], pred)
    
  }
  ### CALCULATE SVM PERFORMANCE ###
  roc <- sum((out[, c(1)] - out[, c(2)])^2) / nrow(out)
  data.frame(parms[i, ], roc)
}

epsilon <- min(result[result[, c(4)] <= (min(result[,c(4)])), c(1)])
cost <- c(-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10)
parms <- expand.grid(epsilon = epsilon, cost = cost, gamma = gamma)
### LOOP THROUGH PARAMETER VALUES ###
result <- foreach(i = 1:nrow(parms), .combine = rbind, .packages = c("foreach", "doParallel", "readr", "dplyr", "assertr", "rsample")) %dopar% {
  c <- parms[i, ]$cost
  g <- parms[i, ]$gamma
  e <- parms[i, ]$epsilon
  ### K-FOLD VALIDATION ###
  out <- foreach(j = 1:(nfolds), .combine = rbind,.packages = c("foreach", "doParallel", "readr", "dplyr", "assertr", "rsample"), .inorder = FALSE) %dopar% {
    deve <- df2[folds[[j]], ]
    test <- na.omit(df2[-folds[[j]], ])    
    mdl <- e1071::svm(ytrain~., data = deve, cost = 2^c, gamma = 2^g, epsilon = 2^e)
    pred <- predict(mdl, test)
    data.frame(test[, c(1)], pred)
    
  }
  ### CALCULATE SVM PERFORMANCE ###
  roc <- sum((out[, c(1)] - out[, c(2)])^2) / nrow(out)
  data.frame(parms[i, ], roc)
}

cost <- min(result[(result[, c(4)] <= (min(result[,c(4)]))), c(2)])
gamma <- c(-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10)
parms <- expand.grid(epsilon = epsilon, cost = cost, gamma = gamma)
### LOOP THROUGH PARAMETER VALUES ###
result <- foreach(i = 1:nrow(parms), .combine = rbind, .packages = c("foreach", "doParallel", "readr", "dplyr", "assertr", "rsample")) %dopar% {
  c <- parms[i, ]$cost
  g <- parms[i, ]$gamma
  e <- parms[i, ]$epsilon
  ### K-FOLD VALIDATION ###
  out <- foreach(j = 1:(nfolds), .combine = rbind,.packages = c("foreach", "doParallel", "readr", "dplyr", "assertr", "rsample"), .inorder = FALSE) %dopar% {
    deve <- df2[folds[[j]], ]
    test <- na.omit(df2[-folds[[j]], ])    
    mdl <- e1071::svm(ytrain~., data = deve, cost = 2^c, gamma = 2^g, epsilon = 2^e)
    pred <- predict(mdl, test)
    data.frame(test[, c(1)], pred)
    
  }
  ### CALCULATE SVM PERFORMANCE ###
  roc <- sum((out[, c(1)] - out[, c(2)])^2) / nrow(out)
  data.frame(parms[i, ], roc)}

gamma <- min(result[(result[, c(4)] <= (min(result[,c(4)]))), c(3)])
bestperformance <- min(result[, c(4)])

#最終的なVI-SVRモデルの構築、テストデータに対する予測精度のチェック
best_model_VI_SVR <- svm(ytrain~., data = xtrain, cost = 2^cost, gamma = 2^gamma, epsilon = 2^epsilon)
train_pred_vi <- predict(best_model_VI_SVR)
test_pred_vi <- predict(best_model_VI_SVR, xtest)

plot(0, 0, type = "n", xlim = c(150, 500), ylim = c(150, 500),xlab = "Observed Value", ylab = "Predicted Value")

points(test_labels, test_pred_vi, col = "orange", pch = 2)
points(train_labels, train_pred_vi, col = "darkgray", pch = 3)
abline(a=0, b=1)

#せっかくなので、途中で作ったランダムフォレストの結果と比較してみる
train_pred_rf <- predict(modelVI)
test_pred_rf <- predict(modelVI, test_data)

plot(0, 0, type = "n", xlim = c(150, 500), ylim = c(150, 500),xlab = "Observed Value", ylab = "Predicted Value")

points(test_labels, test_pred_rf, col = "orange", pch = 2)
points(train_labels, train_pred_rf, col = "darkgray", pch = 3)
abline(a=0, b=1)

#ただのSVRだとどうなるかもやってみる
xtrain_svr <- train_data
xtest_svr <- test_data
ytrain <- train_labels
df2 <- as.data.frame(cbind(ytrain,xtrain_svr))

### PARAMETER LIST ###
cost <- 3
epsilon <- c(-10,-9,-8,-7,-6,-5,-4,-3,-2,-1, 0)

gam <- foreach(k = -20:10, .combine = rbind,.packages = c("kernlab"))%dopar%{
  k = 1
  rbf <- rbfdot(sigma = 2^k)
  rbf
  
  asmat <- as.matrix(xtrain_svr)
  asmat
  
  kern <- kernelMatrix(rbf, asmat)
  sd(kern)
  data.frame((k +21), sd(kern))
}

hakata <- which.max(gam$sd.kern.)

gamma <- hakata - 21
parms <- expand.grid(epsilon = epsilon, cost = cost, gamma = gamma)
### LOOP THROUGH PARAMETER VALUES ###
result <- foreach(i = 1:nrow(parms), .combine = rbind, .packages = c("foreach", "doParallel", "readr", "dplyr", "assertr", "rsample")) %dopar% {
  c <- parms[i, ]$cost
  g <- parms[i, ]$gamma
  e <- parms[i, ]$epsilon
  ### K-FOLD VALIDATION ###
  out <- foreach(j = 1:(nfolds), .combine = rbind,.packages = c("foreach", "doParallel", "readr", "dplyr", "assertr", "rsample"), .inorder = FALSE) %dopar% {
    deve <- df2[folds[[j]], ]
    test <- na.omit(df2[-folds[[j]], ])    
    mdl <- e1071::svm(ytrain~., data = deve, cost = c, gamma = 2^g, epsilon = 2^e)
    pred <- predict(mdl, test)
    data.frame(test[, c(1)], pred)
    
  }
  ### CALCULATE SVM PERFORMANCE ###
  roc <- sum((out[, c(1)] - out[, c(2)])^2) / nrow(out)
  data.frame(parms[i, ], roc)
}

epsilon <- min(result[result[, c(4)] <= (min(result[,c(4)])), c(1)])
cost <- c(-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10)
parms <- expand.grid(epsilon = epsilon, cost = cost, gamma = gamma)
### LOOP THROUGH PARAMETER VALUES ###
result <- foreach(i = 1:nrow(parms), .combine = rbind, .packages = c("foreach", "doParallel", "readr", "dplyr", "assertr", "rsample")) %dopar% {
  c <- parms[i, ]$cost
  g <- parms[i, ]$gamma
  e <- parms[i, ]$epsilon
  ### K-FOLD VALIDATION ###
  out <- foreach(j = 1:(nfolds), .combine = rbind,.packages = c("foreach", "doParallel", "readr", "dplyr", "assertr", "rsample"), .inorder = FALSE) %dopar% {
    deve <- df2[folds[[j]], ]
    test <- na.omit(df2[-folds[[j]], ])    
    mdl <- e1071::svm(ytrain~., data = deve, cost = 2^c, gamma = 2^g, epsilon = 2^e)
    pred <- predict(mdl, test)
    data.frame(test[, c(1)], pred)
    
  }
  ### CALCULATE SVM PERFORMANCE ###
  roc <- sum((out[, c(1)] - out[, c(2)])^2) / nrow(out)
  data.frame(parms[i, ], roc)
}

cost <- min(result[(result[, c(4)] <= (min(result[,c(4)]))), c(2)])
gamma <- c(-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10)
parms <- expand.grid(epsilon = epsilon, cost = cost, gamma = gamma)
### LOOP THROUGH PARAMETER VALUES ###
result <- foreach(i = 1:nrow(parms), .combine = rbind, .packages = c("foreach", "doParallel", "readr", "dplyr", "assertr", "rsample")) %dopar% {
  c <- parms[i, ]$cost
  g <- parms[i, ]$gamma
  e <- parms[i, ]$epsilon
  ### K-FOLD VALIDATION ###
  out <- foreach(j = 1:(nfolds), .combine = rbind,.packages = c("foreach", "doParallel", "readr", "dplyr", "assertr", "rsample"), .inorder = FALSE) %dopar% {
    deve <- df2[folds[[j]], ]
    test <- na.omit(df2[-folds[[j]], ])    
    mdl <- e1071::svm(ytrain~., data = deve, cost = 2^c, gamma = 2^g, epsilon = 2^e)
    pred <- predict(mdl, test)
    data.frame(test[, c(1)], pred)  }
  
  ### CALCULATE SVM PERFORMANCE ###
  roc <- sum((out[, c(1)] - out[, c(2)])^2) / nrow(out)
  data.frame(parms[i, ], roc)
  
  gamma <- min(result[(result[, c(4)] <= (min(result[,c(4)]))), c(3)])
  bestperformance <- min(result[, c(4)])}

#ただのSVRモデルにおけるテストデータに対する予測精度をチェックする
modelSVM <- svm(ytrain~., data = xtrain_svr, cost = 2^cost, gamma = 2^gamma, epsilon = 2^epsilon)
train_pred_svr <- predict(modelSVM)
test_pred_svr <- predict(modelSVM, xtest_svr)

plot(0, 0, type = "n", xlim = c(150, 500), ylim = c(150, 500),xlab = "Observed Value", ylab = "Predicted Value")

points(test_labels, test_pred_svr, col = "orange", pch = 2)
points(train_labels, train_pred_svr, col = "darkgray", pch = 3)
abline(a=0, b=1)
