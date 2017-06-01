setwd("C:/Users/pjha42/OfficeWork/Knowledge&Learning/Flow/Numerai/58/code")

library(pacman)
p_load(data.table)
p_load(randomForest)
p_load(ggplot2)
p_load(caret)
p_load(devtools)
install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
p_load(xgboost)
p_load(Amelia)


train_full <- fread("../data/numerai_training_data.csv", head = T)
test_full <- fread("../data/numerai_tournament_data.csv", head = T)


train_full$target <- as.factor(train_full$target)
test_full$target <- as.factor(test_full$target)

########################################################################################
#### Exploratory Data Analysis ####

## Distribution of classes per era
100*table(train_full$target)/nrow(train_full)

get_eradist <- function(df) {
	era_df <- data.table()
	eras <- unique(df$era)
	for(er in eras){
		temp_data <- df[era %in% er, ]
		x <- as.numeric(100*table(temp_data$target)/nrow(temp_data))
		era_df <- rbind(era_df, cbind(era = er, factor_0 = x[1], factor_1 = x[2]))
	}
	era_df[, diff:= as.numeric(factor_0) - as.numeric(factor_1),]
	return(era_df)
}

train_era <- get_eradist(train_full)
test_era <- get_eradist(test_full)

## Ploting

plot(train_era$diff, type = 'l')
plot(test_era$diff, type = 'l')

missmap(train_full[, grep("feature|target", names(train_full)), with=F])


#############################################################################
## PCA for dimensionality reduction

train_pca <- prcomp(train_full[, grep("feature", names(train_full)), with=F], scale = T)

std_dev <- train_pca$sdev
pca_var <- std_dev^2
prop_varex <- pca_var/sum(pca_var)
sum(prop_varex[1:6])

# Scree Plot
plot(prop_varex, xlab = "Principal Component",
	ylab = "Proportion of Variance Explained",
	type = "b")

## Number of principal components selected - 6


##################################################################################
#### Predictive Modeling ####

## Data prep according to PCA ##
train_data <- data.table(era=train_full$era, target=train_full$target, train_pca$x[,1:6])
test_data <- data.frame(era=test_full$era, data_type = test_full$data_type, target = test_full$target, predict(train_pca, newdata=test_full))
test_data <- data.table(test_data[,1:9])
validation_data <- test_data[data_type=='validation',]


#### XGBoost Model Building ####

#convert data frame to data table
train <- train_data
test <- validation_data
setDT(train) 
setDT(test)

#using one hot encoding 
labels <- train$target 
ts_label <- test$target
new_tr <- model.matrix(~.+0,data = train[,-c("target", "era"),with=F]) 
new_ts <- model.matrix(~.+0,data = test[,-c("target", "era", "data_type"),with=F])

#convert factor to numeric 
labels <- as.numeric(labels)-1
ts_label <- as.numeric(ts_label)-1
table(labels)
table(ts_label)

#preparing matrix 
dtrain <- xgb.DMatrix(data = new_tr,label = labels) 
dtest <- xgb.DMatrix(data = new_ts,label=ts_label)

#default parameters
params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.0001, 
		gamma=5, max_depth=3, min_child_weight=1, subsample=0.5, colsample_bytree=0.4)


## model training
xgb1 <- xgb.train (params = params, data = dtrain, nrounds = 200, 
		watchlist = list(val=dtest,train=dtrain), 
		print_every_n = 10, early_stop_round = 50, 
		maximize = F , eval_metric = "logloss")

# model prediction
p_load(Metrics)
xgbpred <- predict (xgb1,dtest)
#logLoss(ts_label, xgbpred)
get_logloss(validation_data, ts_label, xgbpred)


## era wise log loss
get_logloss <- function(df, actual, prediction) {
	df2 <- cbind(df, actual, prediction)
	eras <- unique(df2$era)
	sum = 0;
	for(er in eras) {
		temp_df <- df2[era == er, ]
		ll <- logLoss(temp_df$actual, temp_df$prediction)
		#cat("\nEra: ", er, " Log-Loss: ", ll, "	status:", ifelse(ll < (-log(0.5)), 1, 0))
		sum = sum + ifelse(ll < (-log(0.5)), 1, 0)
	}
	#cat("\nSum: ", sum, " Prop: ", 100*(sum/length(eras)))
	return(100*(sum/length(eras)))
}
get_logloss(validation_data, ts_label, xgbpred)

#train data era-wise log loss
xgbpred_tr <- predict (xgb1,dtrain)
get_logloss(train_data, labels, xgbpred_tr)


xgbpred <- ifelse (xgbpred > 0.5,1,0)

p_load(caret)
p_load(e1071)
confusionMatrix (xgbpred, ts_label)
#Accuracy - 86.54%

#view variable importance plot
mat <- xgb.importance (feature_names = colnames(new_tr),model = xgb1)
xgb.plot.importance (importance_matrix = mat) 



#########################################################################
## Manual parameter tuning ##
eta = c(0.000005, 0.00001, 0.00005, 0.0001, 0.001, 0.01, 0.1)
gamma = c(0, 2, 3, 5)
rounds = c(15, 30, 50, 100) 

eval_result <- data.table()
for(i in eta){
	for(j in gamma){
	for(k in rounds){
		cat("Eta: ", i, " Gamma: ", j, "Rounds: ", k, "\n")
		params <- list(booster = "gbtree", objective = "binary:logistic", eta=i, 
			gamma=j, max_depth=3, min_child_weight=1, subsample=0.5, colsample_bytree=0.4)
		## model training
		xgb1 <- xgb.train (params = params, data = dtrain, nrounds = k, 
			watchlist = list(val=dtest,train=dtrain), 
			print_every_n = 10, early_stop_round = 20, 
			maximize = F , eval_metric = "logloss")

		# model prediction
		xgbpred <- predict (xgb1,dtest)
		#
		consistency <- get_logloss(validation_data, ts_label, xgbpred)
		eval_result <- rbind(eval_result, cbind(ETA = i, Gamma = j, Nrounds = k, consistency, logloss = logLoss(ts_label, xgbpred)))
	}
	}
}

View(eval_result)

# Best parameters: nrounds - 30, logloss - 0.692, gamma - 3

#### Hyper-parameter tuning ####
xgb_grid_1 <- expand.grid(
	gamma = c(0, 5, 10, 15),
	nrounds = 200,
	eta = c(0.0001, 0.001, 0.01, 0.1),
	max_depth = c(3, 4, 5),
	subsample = c(0.5, 0.7),
	colsample_bytree = c(0.5, 0.8),
	min_child_weight = 1
)

# pack the training control parameters
xgb_trcontrol_1 = trainControl(
  method = "cv",
  number = 4,
  verboseIter = TRUE,
  returnData = FALSE,
  returnResamp = "all",                  # save losses across all models
  classProbs = F,
  summaryFunction = twoClassSummary,
  allowParallel = F
)


# train the model for each parameter combination in the grid, 
#   using CV to evaluate
xgb_train_1 = train(
  x = as.matrix(train_data[, grep("PC", names(train_data)), with = F]),
  y = train_data$target,
  trControl = xgb_trcontrol_1,
  tuneGrid = xgb_grid_1,
  method = "xgbTree"
)

	






# training and validation set
train_data <- train_full[1:85549,]
validation_data <- train_full[85550:nrow(train_full),]

train <- train_data[, 4:25]
validation <- validation_data[, 4:24]


# fitting basic random forest model
fit_rf1 <- randomForest(target~., train, ntree = 200, mtry = 5, importance = T)
summary(fit_rf1)

predicted <- predict(fit_rf1, validation)


seed <- 123

# Algorithm Tune (tuneRF)
set.seed(seed)
bestmtry <- tuneRF(train[,1:21], train$target, stepFactor=1, improve=1e-5, ntree=100)
print(bestmtry)

metric <- "Accuracy"
# caret - grid search - random forest
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
set.seed(seed)
tunegrid <- expand.grid(.mtry=c(3:6))
rf_gridsearch <- train(target~., data=train, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_gridsearch)
plot(rf_gridsearch)


# Manual Search
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
tunegrid <- expand.grid(.mtry=c(sqrt(ncol(train))))
modellist <- list()
for (ntree in c(50, 100, 150, 200)) {
	set.seed(seed)
	print(ntree)
	fit <- train(target~., data=train, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control, ntree=ntree)
	key <- toString(ntree)
	modellist[[key]] <- fit
}
# compare results
results <- resamples(modellist)
summary(results)
dotplot(results)




