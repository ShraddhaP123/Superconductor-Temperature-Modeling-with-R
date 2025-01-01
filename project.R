# Load required libraries
library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)

# Read the data
train_data <- read.csv("train.csv")
unique_m_data <- read.csv("unique_m.csv")

# Inspect the data
glimpse(train_data)
glimpse(unique_m_data)

# Check for missing values
sum(is.na(train_data))
sum(is.na(unique_m_data))

# Data preprocessing
train_data <- na.omit(train_data)  # Remove rows with missing values if necessary

# Split data into features and target variable
X <- as.matrix(train_data[, 1:81])  # Convert features to matrix
y <- train_data[, 82]  # Critical temperature (target variable)

# Train-test split
set.seed(123)
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[trainIndex, ]
X_test <- X[-trainIndex, ]
y_train <- y[trainIndex]
y_test <- y[-trainIndex]

# Model 1: Linear Regression with Cross-Validation
cv_control <- trainControl(method = "cv", number = 5)
lm_model <- train(X_train, y_train, method = "lm", trControl = cv_control)

# Model 2: Random Forest with Hyperparameter Tuning
rf_tune <- train(
  X_train, y_train,
  method = "rf",
  tuneGrid = expand.grid(.mtry = c(10, 20, 30)),
  trControl = trainControl(method = "cv", number = 5),
  ntree = 500
)
rf_model <- rf_tune$finalModel

# Model 3: XGBoost with Hyperparameter Tuning
xgb_grid <- expand.grid(
  nrounds = c(200, 500, 1000),
  max_depth = c(4, 6, 8),
  eta = c(0.01, 0.1, 0.3),
  gamma = c(0, 0.1, 0.2),
  colsample_bytree = c(0.6, 0.8, 1),
  min_child_weight = c(1, 3, 5),
  subsample = c(0.7, 0.8, 1)
)

xgb_tune <- train(
  X_train, y_train,
  method = "xgbTree",
  tuneGrid = xgb_grid,
  trControl = trainControl(method = "cv", number = 5)
)
xgb_model <- xgb_tune$finalModel

# Predictions
lm_predictions <- predict(lm_model, X_test)
rf_predictions <- predict(rf_model, X_test)
xgb_predictions <- predict(xgb_model, X_test)

# Model Evaluation
lm_rmse <- sqrt(mean((lm_predictions - y_test)^2))
rf_rmse <- sqrt(mean((rf_predictions - y_test)^2))
xgb_rmse <- sqrt(mean((xgb_predictions - y_test)^2))

cat("Linear Regression RMSE:", lm_rmse, "\n")
cat("Random Forest RMSE:", rf_rmse, "\n")
cat("XGBoost RMSE:", xgb_rmse, "\n")

rmse_results <- paste(
  "Linear Regression RMSE:", lm_rmse, "\n",
  "Random Forest RMSE:", rf_rmse, "\n",
  "XGBoost RMSE:", xgb_rmse, "\n"
)
writeLines(rmse_results, "rmse_results.txt")
