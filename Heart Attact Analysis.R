# According to the World Health Organization, a terminal sickness is one that has a steady 
# progression towards Heart disease and is currently the world's deadliest disease.

#According to data, 68 percent of deaths were caused by chronic long-term diseases, with 
#heart disease being the leading cause.

# As a Entry Level data scientist, I have use various predictor factors to try to generate a 
#forecast about heart disease patients.

# Also,I have use Logistic regression and K-Nearest Neighbor to develop a model to predict whether 
# the patients have heart disease or not for the analysis.


# Import Library

library(dplyr) # Data Manipulation
library(gtools) # Handling R Packages
library(gmodels) # Mean,bi-normal proportion & probability
library(ggplot2) # Data visualization
library(class) # Multidimensional Scaling for prediction,confusion matrix & K-N-N
library(tidyr) # Data Sorting
library(lattice) # Data visualization
library(caret) # Regression
library(e1071) # If you install caret sometimes R gives warring so better install it
library(rmdformats) # Ready to use HTML output

# Import Heart Data

getwd()
setwd("/Users/kushtrivedi/Downloads")
heartData <- read.csv("heart.csv")

# Explore Overview of Data
summary(heartData)
head(heartData)
tail(heartData)
glimpse(heartData)


# Note: 
#     age:      Age of Patient
#     sex:      0 = Female & 1 = Male
#     cp:      Levels of Chest Pain (0-3)
#     trtbps:   Blood Pressure at Resting (mm HG)
#     fbs:      blood sugar:- 1 means above 120 mg/dl & 0 means below 120 mg/dl
#     restecg:  result of electrocardiograph
#     thalachh: Maximum Heartbeat Rate
#     exng:     exercised include angina:- 1 means YES & 0 means NO
#     oldpeak:  Exercise Relative to Rest
#     slp:      Slope
#     caa:      Number of the most blood vessel
#     thall:    Form of thalassemia
#     output:   0 means NO-SICKNESS & 1 means Sickness


# Data Wrangling ( often used for changing data to its correct type)

heartData <- heartData %>%
  mutate(cp = as.factor(cp),
         restecg = as.factor(restecg),
         slp = as.factor(slp),
         caa = as.factor(caa),
         thall = as.factor(thall),
         sex = factor(sex, levels = c(0,1), labels = c("female", "male")),
         fbs = factor(fbs, levels = c(0,1), labels = c("False", "True")),
         exng = factor(exng, levels = c(0,1), labels = c("No", "Yes")),
         output = factor(output, levels = c(0,1), labels = c("Health", "Not Health")))
# Cross verify
head(heartData)
glimpse(heartData)

# Check for Missing Value (In this case there no Missing Value)
colSums(is.na(heartData))



# Data Pre-Processing (Mind-Map the model by checking the proportion for required field)

prop.table(table(heartData$sex)) # Proportion
table(heartData$sex) # Total (96 + 207) = 303 Cross- Verification
prop.table(table(heartData$output)) # Proportion
table(heartData$output) # Total (138 + 165) = 303 Cross- Verification



#Cross Validation (Try and Test) and after testing we will use that data to visualization

set.seed(101)
index <- sample(nrow(heartData),nrow(heartData) * 0.7)

# Data Try
try_Heart_Data <- heartData[index,]
# Data Test
test_Heart_Data <- heartData[-index,]

# Lets Check weather TEST Data is able to fir our model if not we will 
# need to do cross validation again
prop.table(table(heartData$output)) # Without Cross validation
prop.table(table(try_Heart_Data$output)) # With Cross validation


# Data Modelling

# Create a Model by using TRY model and play with important variables such as sex,output,cp etc...
heart_Model_1 <- glm(formula = output ~ sex + cp + fbs + thall, 
                     family = "binomial", data = try_Heart_Data)
# "binomial" is a function and will return Deviance, Coefficients and Dispersion shown bellow
summary(heart_Model_1)

# By watching it carefully some of variables are not necessary for our model so we create all
# variables step by step and will create a better model for that there is simple way to do
# that see bellow

# Model without Predictor
heart_Model_NoPredictor <- glm(output ~ 1, family = "binomial", data = try_Heart_Data)
# Model without Predictor
heart_Model_AllPredictor <- glm(output ~ ., family = "binomial", data = try_Heart_Data)


# Step wise regression Forward, Backward & Both for All Predictor Heart Model
# Backward
heart_Model_Backward <- step(object = heart_Model_AllPredictor, 
                             direction = "backward", trace = F)
# Forward
heart_Model_Forward <- step(object = heart_Model_AllPredictor, 
                            scope = list(lower = heart_Model_NoPredictor, 
                                         upper = heart_Model_AllPredictor), 
                            direction = "forward",trace = F)
# Backward & Forward Both
heart_Model_BackwardForward <- step(object = heart_Model_AllPredictor,
                                    scope = list(lower = heart_Model_NoPredictor,
                                                 upper = heart_Model_AllPredictor),
                                    direction = "both",trace = F)

# Lets See summary of each model

# Backward and Backward-Forward has same Residual Deviance = 122.59 and AIC = 156.59
# While forward has  RD = 120.23 and AIC = 166.23
summary(heart_Model_Backward)
summary(heart_Model_Forward)
summary(heart_Model_BackwardForward)


# Prediction

# So lets predict the model by using Backward-Forward and will use TEST data to predict

test_Heart_Data$prediction <- predict(heart_Model_BackwardForward, 
                                      type = "response", 
                                      newdata = test_Heart_Data)
# Create Plot for prediction

test_Heart_Data %>% 
  ggplot(aes(x=prediction)) + 
  geom_density() +
  labs(title = "Prediction Data Probabilities") +
  theme_gray()
# According to result it inclines more towards 1 in output(column of table) 
# which means (Not Health)

# Lets have more clear view by comparing it
prediction_DataFrame <- predict(heart_Model_BackwardForward, 
                               type = "response", 
                               newdata = test_Heart_Data)
result_Prediction_DataFrame <- ifelse(prediction_DataFrame >= 0.5,
                                      "Not Health","Health")

#Override our result_Prediction_Data-Frame to test_Heart_Data$prediction
test_Heart_Data$prediction <- result_Prediction_DataFrame

#Overview of comparison(More occurrence of Not Health) try using different head and see
test_Heart_Data %>%
  select(output, prediction) %>%
  head(12)

# Model Evaluation (how good our model had predict: Accuracy (able to predict sensitivity) 
#  & Specificity(able to predict precision)  by creating confusion matrix)
confusion_Matrix <- confusionMatrix(as.factor(result_Prediction_DataFrame), 
                                    reference = test_Heart_Data$output,
                                    positive = "Not Health")
confusion_Matrix 

# Not Health of Not Health = 42
# Health of Not Health = 5
# Health of Health = 35
#Not Health of Health = 9
# Lest make that it more understandable
recall <- round(42/(42+5),3)
specificity <- round(35/(35+9),3)
precision <- round(42/(42+11),3)
accuracy <- round((42+35)/(42+35+9+5),3)

matrix <- cbind.data.frame(accuracy,recall,specificity,precision)
# More Clear Result
matrix

# Model Interpretation: 
# Transforming odd value probabilities and analyzing coefficient of model to see 
# Positive class probability

# Probability
heart_Model_BackwardForward$coefficients %>%
  inv.logit() %>% # Transform odd values
  data.frame()

# Prediction 1: Males have a 18.8 percent chance of being diagnosed with heart disease.
# Prediction 2: People with a high level of severe pain (cp = 3) have a 90 percent chance of 
#               developing heart disease.



# K - Nearest Neighbor 

# Data Wrangling : as we will use K - Nearest Neighbor in order to that we will create 
# new data frame consisting of dummy variable that we will predict to our output variable

dummy_DataFrame <- dummyVars("~output + sex + cp + trtbps + chol + fbs + restecg + thalachh + 
                             exng + oldpeak + slp + caa + thall", data = heartData)

# Create new data frame
dummy_DataFrame <- data.frame(predict(dummy_DataFrame,newdata = heartData))

# Let's check structure of dummy data frame
str(dummy_DataFrame)


dummy_DataFrame$output.Health <- NULL
dummy_DataFrame$sex.female <- NULL
dummy_DataFrame$fbs.False <- NULL
dummy_DataFrame$exng.No <- NULL

head(dummy_DataFrame)

# Cross Validation: K - Nearest Method
# it has different approach in compare to Logistic regression
# Here we will split Predictor and output variable(column in heart table) into try & test

set.seed(101)

# Predictor
try_Predictor_Knn_Dummy_Heart_Data_X <- dummy_DataFrame[index, -1]
test_Predictor_Knn_Dummy_Heart_Data_X <- dummy_DataFrame[-index, -1]
# Output
try_Predictor_Knn_Dummy_Heart_Data_Y <- dummy_DataFrame[index, 1]
test_Predictor_Knn_Dummy_Heart_Data_Y <- dummy_DataFrame[-index, 1]


# Choose k by a common method: square root of data count
sqrt(nrow(try_Predictor_Knn_Dummy_Heart_Data_X)) 
# K will be 14.56044 = 14 and will use for prediction in next step


# Create K - Nearest Neighbor Prediction
prediction_Knn <- knn(train = try_Predictor_Knn_Dummy_Heart_Data_X,
                      test = test_Predictor_Knn_Dummy_Heart_Data_X,
                      cl = try_Predictor_Knn_Dummy_Heart_Data_Y,
                      k = 14)

# We will Transform Knn prediction into data frame and rename to orignal label

prediction_Knn <- prediction_Knn %>%
  as.data.frame() %>%
  mutate(prediction_Knn = factor(prediction_Knn,
                                 levels = c(0,1),
                                 labels = c("Health", "Not Health")))%>%select(prediction_Knn)
# Same with confusion matrix
test_Predictor_Knn_Dummy_Heart_Data_Y <- test_Predictor_Knn_Dummy_Heart_Data_Y %>%
  as.data.frame() %>%
  mutate(output = factor(test_Predictor_Knn_Dummy_Heart_Data_Y,
                                 levels = c(0,1),
                                 labels = c("Health", "Not Health")))%>%select(output)


# Create Confusion Matrix
confusion_Matrix_Knn <- confusionMatrix(prediction_Knn$prediction_Knn, 
                                        reference = test_Predictor_Knn_Dummy_Heart_Data_Y$output,
                                        positive = "Not Health")
confusion_Matrix_Knn

# Not Health of Not Health = 37
# Health of Not Health = 10
# Health of Health = 29
#Not Health of Health = 15
# Lest make that it more understandable
recall_Knn <- round(37/(37+10),3)
specificity_Knn  <- round(29/(29+15),3)
precision_Knn  <- round(37/(37+15),3)
accuracy_Knn  <- round((37+29)/(37+29+15+10),3)

matrix_Knn <- cbind.data.frame(accuracy_Knn,recall_Knn,specificity_Knn,precision_Knn)
# More Clear Result
matrix_Knn

# Overall Prediction of K-N-N Model is 72.5 % accuracy
# Not Health person are 78 %
# Health person are 65.9 %
# Precision for "Not Health" is 71.2 % from our K-N-N Prediction


# Compare Logistic Regression and K-N-N

#LR
matrix
#K-N-N
matrix_Knn

# We can see that K-N-N has better specificity and precision in compare to Logistic Regression
# Logistic Regression has  better accuracy and recall 


#Conclusion:
#   If a doctor has to chose treat people with heart disease differently, we would go with 
#   better precision.
#   if a doctor merely wanted to diagnose as many people as possible with heart disease while 
#   ignoring the incorrect categorization, then we will use best recall.



# Q-Plot Scatter for Age and Cholesterol > 200
heart_Cholesterol_DataFrame_Above_200 <- heartData[heartData$chol > 200,]
qplot(data = heart_Cholesterol_DataFrame_Above_200, x = age, y= chol,colour= age, size=I(5),alpha = I(0.7),main = "Age of people where Cholesterol > 200",xlab = "Age",ylab = "Cholesterol")

#  Q-Plot Scatter for Age and Cholesterol < 200
heart_Cholesterol_DataFrame_Below_200 <- heartData[heartData$chol < 200,]
qplot(data = heart_Cholesterol_DataFrame_Below_200, x = age, y= chol,colour= age, size=I(5),alpha = I(0.7),main = "Age of people where Cholesterol > 200",xlab = "Age",ylab = "Cholesterol")


#Q-Plot Box plot for Chest pain level below age of 50
heart_CestPain_Data_Age_Below_50 <- heartData[heartData$age < 50,]
qplot(data = heart_CestPain_Data_Age_Below_50,x = cp, y = age,colour = cp,size=I(1),alpha = I(0.7), geom = "boxplot",main = "Chest Pain level of people under age of 50.",xlab = "Chest Pain", ylab = "Age")
#Q-Plot Box plot for Chest pain level above age of 50
heart_CestPain_Data_Age_Above_50 <- heartData[heartData$age > 50,]
qplot(data = heart_CestPain_Data_Age_Above_50,x = cp, y = age,colour = cp,size=I(1),alpha = I(0.7), geom = "boxplot",main = "Chest Pain level of people above age of 50.",xlab = "Chest Pain", ylab = "Age")














