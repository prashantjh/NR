source("./import_data.R")

#### Predictive Modeling ####

## Data prep according to PCA ##
train_data <- data.table(era=train_full$era, target=train_full$target, 
				train_pca$x[,1:6])
test_data <- data.frame(era=test_full$era, data_type = test_full$data_type, 
				target = test_full$target,
				predict(train_pca, newdata=test_full))
test_data <- data.table(test_data[,1:9])

validation_data <- test_data[data_type=='validation',]

p_load(Metrics)
## era wise log loss
get_logloss <- function(df, actual, predicted) {
	df2 <- cbind(df, actual, predicted)
	eras <- unique(df2$era)
	sum = 0;
	for(er in eras) {
		temp_df <- df2[era == er, ]
		ll <- logLoss(temp_df$actual, temp_df$predicted)
		cat("\nEra: ", er, " Log-Loss: ", ll, "	status:", ifelse(ll < (-log(0.5)), 1, 0))
		sum = sum + ifelse(ll < (-log(0.5)), 1, 0)
	}
	cat("\nSum: ", sum, " Prop: ", 100*(sum/length(eras)))
	return(100*(sum/length(eras)))
}


#### Model Building ####
glm_model <- glm(formula = target ~., family = binomial(link = 'logit'),
				data = train_data[, 2:8]) 




#validation data era-wise log loss
model <- glm_model
prediction <- predict (model, validation_data[, 4:9], type='response')
classification <- ifelse (prediction > 0.5037, 1, 0)

actual <- as.numeric(validation_data$target)-1
get_logloss(validation_data, actual, prediction)



p_load(caret)
p_load(e1071)
confusionMatrix (classification, actual)

## Test data prediction
test_pred <- predict (model, test_data[, 4:9], type='response')

test_out <- cbind(id = test_full$id, probability = test_pred)
write.csv(test_out, "../results/2_glm.csv", row.names = F)