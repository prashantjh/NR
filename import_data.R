library(pacman)
p_load(data.table)
p_load(randomForest)
p_load(ggplot2)
p_load(caret)
p_load(devtools)
drat:::addRepo("dmlc")
p_load(xgboost)
p_load(Amelia)
p_load(Metrics)
p_load(e1071)
p_load(kernlab)

train_full <- fread("../data/numerai_training_data.csv", head = T)
test_full <- fread("../data/numerai_tournament_data.csv", head = T)


train_full$target <- as.factor(train_full$target)
test_full$target <- as.factor(test_full$target)


#############################################################################
## PCA for dimensionality reduction

train_pca <- prcomp(train_full[, grep("feature", names(train_full)), with=F], scale = T)

std_dev <- train_pca$sdev
pca_var <- std_dev^2
prop_varex <- pca_var/sum(pca_var)

#### Data prep according to PCA ####
train_data <- data.table(era=train_full$era, target=train_full$target, 
				train_pca$x[,1:6])
test_data <- data.frame(era=test_full$era, data_type = test_full$data_type, 
				target = test_full$target,
				predict(train_pca, newdata=test_full))
test_data <- data.table(test_data[,1:9])

validation_data <- test_data[data_type=='validation',]

## era wise log loss
get_logloss <- function(df, actual, predicted) {
	df2 <- cbind(df, actual, predicted)
	eras <- unique(df2$era)
	sum = 0;
	for(er in eras) {
		temp_df <- df2[era == er, ]
		ll <- logLoss(temp_df$actual, temp_df$predicted)
		#cat("\nEra: ", er, " Log-Loss: ", ll, "	status:", ifelse(ll < (-log(0.5)), 1, 0))
		sum = sum + ifelse(ll < (-log(0.5)), 1, 0)
	}
	cat("\nSum: ", sum, " Prop: ", 100*(sum/length(eras)))
	return(100*(sum/length(eras)))
}

