diamonds_cleaned <- read.csv("diamonds_cleaned.csv") # Reading the data set as a dataframe
dim(diamonds_cleaned) # Print dimensions of the dataset

# Load library
library(ggplot2)
library(GGally)
# Select the right-skewed and related variables
selected_vars <- diamonds_cleaned[, c("price", "carat", "x", "y", "z")]
# Visualize pairwise relationships between the selected variables
ggpairs(selected_vars[sample(53738, 1000), ])

# Create the new variable 'average_size'
diamonds_cleaned$average_size <- with(diamonds_cleaned, (x + y + z) / 3)

# Create the new variable 'price_per_carat'
diamonds_cleaned$price_per_carat <- with(diamonds_cleaned, price/carat)

# Select the right-skewed and related variables
selected_vars_new <- diamonds_cleaned[, 
c("price_per_carat", "carat", "average_size", "depth", "table")]
# Visualize pairwise relationships between the selected variables
ggpairs(selected_vars_new[sample(53738, 1000), ])

library(gridExtra)

sa <- diamonds_cleaned[sample(53738, 5000), ]
# Create individual box plots
p1 <- ggplot(sa, aes(x = cut, y = price, fill = cut)) + 
geom_boxplot() + labs(x = "Cut", y = "Price") + theme(legend.position = "none")
p2 <- ggplot(sa, aes(x = color, y = price, fill = color)) + 
geom_boxplot() + labs(x = "Color", y = "Price") + theme(legend.position = "none")
p3 <- ggplot(sa, aes(x = clarity, y = price, fill = clarity)) + 
geom_boxplot() + labs(x = "Clarity", y = "Price") + theme(legend.position = "none")
p4 <- ggplot(sa, aes(x = cut, y = price_per_carat, fill = cut)) + 
geom_boxplot() + labs(x = "Cut", y = "Price/Carat") + theme(legend.position = "none")
p5 <- ggplot(sa, aes(x = color, y = price_per_carat, fill = color)) + 
geom_boxplot() + labs(x = "Color", y = "Price/Carat") + theme(legend.position = "none")
p6 <- ggplot(sa, aes(x = clarity, y = price_per_carat, fill = clarity)) + 
geom_boxplot() + labs(x = "Clarity", y = "Price/Carat") + theme(legend.position = "none")
grid.arrange(p1, p4, p2, p5, p3, p6, ncol = 2, nrow = 3, heights = c(3, 3, 3))

# Calculate the median of price_per_carat
median_p <- median(diamonds_cleaned$price_per_carat)
# Create a binary variable (1 if price_per_carat > median, 0 otherwise)
diamonds_cleaned$expensive <- ifelse(diamonds_cleaned$price_per_carat > median_p, 1, 0)
diamonds <- diamonds_cleaned[, c("expensive", "carat", "depth", "table", 
                            "cut", "color", "clarity", "average_size","price_per_carat")]
head(diamonds,n=1) # Display the first row of the modified dataset

# Convert categorical variables to factors
diamonds$cut <- as.factor(diamonds$cut)
diamonds$color <- as.factor(diamonds$color)
diamonds$clarity <- as.factor(diamonds$clarity)
# Split the data into training (60%) and testing (40%) sets
set.seed(12345)
train_index <- sample(seq_len(nrow(diamonds)), size = 0.6 * nrow(diamonds))
train_data <- diamonds[train_index, ]; test_data <- diamonds[-train_index, ]

# Build a logistic regression model
model_logit <- glm(expensive ~ carat + depth + table + cut + color + clarity 
                   + average_size, family = binomial, data = train_data)
# Summary of the logistic model
summary(model_logit)
# Predict on the test data
pred_logit <- predict(model_logit, newdata = test_data, type = "response")
# Convert probabilities to binary predictions (0 or 1)
pred_logit_class <- ifelse(pred_logit > 0.5, 1, 0)
# Confusion matrix for logistic regression
table(Predicted = pred_logit_class, Actual = test_data$expensive)
# Accuracy of the logistic model
accuracy_logit <- mean(pred_logit_class == test_data$expensive)
print(paste("Logistic Regression Accuracy:", accuracy_logit))

library(e1071)
# Build a SVM model for classification model
model_svm <-svm(expensive ~ carat + depth + table + cut + color + clarity + average_size, 
data=train_data, type='C-classification', kernel='radial', cost=.1, gamma=1, scale=TRUE)
testPred <- predict(model_svm, test_data, type="response") # Predict on the test data
print(mean(testPred == test_data$expensive)) # Accuracy

library(doParallel)
library(foreach)
# Detect number of cores
numCores <- detectCores()
cl <- makeCluster(numCores)
registerDoParallel(cl)
# Define grid of parameters to test
cost_values <- c(0.1, 1, 10)
gamma_values <- c(0.1, 1, 10)
# Perform parallel SVM training using foreach
results <- foreach(cost = cost_values, .combine = rbind, .packages = "e1071") %:%
           foreach(gamma = gamma_values, .combine = rbind) %dopar% {
  # Perform hyperparameter tuning using the tune function
  model_svm <- svm(expensive ~ carat + depth + table + cut + color + clarity + 
    average_size, data = train_data,  type='C-classification', 
    kernel = "radial", cost = cost, gamma = gamma, scale = TRUE)
  # Predict on test data using the best model
  test_predictions <- predict(model_svm, test_data, type="response")
  # Calculate accuracy on test data
  accuracy <- mean(test_predictions == test_data$expensive)
  # Return cost, gamma, and accuracy for each combination
  c(cost, gamma, accuracy)
}
# Stop the parallel
stopCluster(cl)
# Print the tuning results
print(results)

# Build a SVM model for classification model
model_svm <-svm(expensive ~ carat + cut + color + clarity + average_size, data=train_data, 
type='C-classification',kernel='radial',cost=10,gamma=0.1,scale=TRUE,probability = TRUE)
print(model_svm)
testPred <- predict(model_svm, test_data, type="response") # Predict on the test data
table(Predicted = testPred, Actual = test_data$expensive) # Confusion matrix
print(mean(testPred == test_data$expensive)) # Accuracy

library(ROCR)
library(MASS)
# Set plot to be square
par(pty = "s") 
# Define custom function for plotting ROC curves
rocPlot <- function(pred, truth, ...) {
   predob <- prediction(pred, truth) # transform input into standardized format
   perf <- performance(predob , "tpr", "fpr") # evaluate predictor performance    
   return(plot(perf, ...))
}
# Logistic Regression - retrieve predicted probabilities for ROC curve
decValues_logit <- predict(model_logit, test_data, type = "response")
rocPlot(decValues_logit, test_data$expensive, col = 'red')
# Get predicted probabilities for SVM
pred_svm <- predict(model_svm, test_data, probability = TRUE)
# Extract probabilities for class '1'
svm_probabilities <- attr(pred_svm, "probabilities")[,1]
rocPlot(svm_probabilities, test_data$expensive, col = 'blue', add = TRUE)
# Add legend to distinguish curves
legend("bottomright", legend = c("Logistic Regression", "SVM"), col = c("red", "blue"))



