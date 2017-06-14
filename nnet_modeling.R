setwd("C:/Flow/Numerai/59/code")
source("import_data.R")
p_load(nnet)

#### Neural Network ####

## Basic NNET ##

#h = 2
#iter = 50

for(h in 2:8) {
	for(iter in c(50, 80, 100, 120, 150, 200, 250)) {
set.seed(123)
cat(h, "\t", iter)
model <- nnet(target~., data=train_data[, 2:8], size = h,
		decay=5e-4, maxit=iter, trace = F)

# Prediction #
actual <- as.numeric(validation_data$target)-1
predicted <- as.numeric(predict(model, validation_data[,4:9]))
get_logloss(validation_data, actual, predicted)
classification <- ifelse(predicted > 0.5, 1, 0)
print(100*mean(actual == classification))
print(logLoss(actual, predicted))
}
}

#Validation Log Loss -  0.6918924
#Training Log Loss - 0.6918235

# Classification #
test_pred <- as.numeric(predict(model, test_data[,4:9]))
test_out <- cbind(id = test_full$id, probability=test_pred)
write.csv(test_out, '../results/5_ann.csv', row.names = F)