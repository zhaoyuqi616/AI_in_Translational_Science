# Load data
BRCA_PAM50_Expression<-read.table("BRCA_PAM50_Expression.txt",sep = ",",header = TRUE)
colnames(BRCA_PAM50_Expression)<-gsub("\\.","-",colnames(BRCA_PAM50_Expression))
# Make sure the samples are identical between the two data set
BRCA_Subtypes$Patients==colnames(BRCA_PAM50_Expression)
BRCA_Subtypes<-read.table("BRCA_Subtypes.txt",sep = ",",header = TRUE)
Subtype_Biomarkers<-c("ERBB2","ESR1","PGR")
Biomarkers_Expression<-BRCA_PAM50_Expression[Subtype_Biomarkers,]
Biomarkers_Expression<-as.data.frame(t(Biomarkers_Expression))
Biomarkers_Expression$Subtypes<-BRCA_Subtypes$Subtypes
Biomarkers_Expression[,1:3]<-log1p(Biomarkers_Expression[,1:3])
Biomarkers_Expression$Subtypes<-as.factor(Biomarkers_Expression$Subtypes)

library(caret)
# create a list of 80% of the rows in the original dataser for training

training_index <- createDataPartition(Biomarkers_Expression$Subtypes, p=0.80, list=FALSE)
# 80% for training 
Training_Data<-Biomarkers_Expression[training_index,]
# 20% for validation
Validation_Data <- Biomarkers_Expression[-training_index,]

# list types for each attribute
sapply(Training_Data, class)
# take a peek at the first 5 rows of the data
head(Training_Data)
# list the levels for the class
levels(Training_Data$Subtypes)
# summarize the class distribution
percentage <- prop.table(table(Training_Data$Subtypes)) * 100
cbind(freq=table(Training_Data$Subtypes), percentage=percentage)
# summarize attribute distributions
summary(Training_Data)
# split input and output
x <- Training_Data[,1:3]
y <- Training_Data[,4]
colors<-c("darkred","lightblue","lightgreen")
par(mfrow=c(1,3))
for(i in 1:3) {
  boxplot(x[,i], main=colnames(Training_Data)[i],col = color[i])
}
# barplot for class breakdown
plot(y,col =1:5)
# scatterplot matrix
featurePlot(x=x, y=y, plot="ellipse",no.legend = FALSE)
# box and whisker plots for each attribute
featurePlot(x=x, y=y, plot="box")

# density plots for each attribute by class value
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)

# Run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

# Linear Discriminant Analysis (lda)
set.seed(600)
fit.lda <- train(Subtypes~., data=Training_Data, method="lda", metric=metric, trControl=control)

# Classification And Regression Tree (cart)
set.seed(600)
fit.cart <- train(Subtypes~., data=Training_Data, method="rpart", metric=metric, trControl=control)

# k-nearest neighbors (knn)
set.seed(600)
fit.knn <- train(Subtypes~., data=Training_Data, method="knn", metric=metric, trControl=control)

# Support vector machines (svm)
set.seed(600)
fit.svm <- train(Subtypes~., data=Training_Data, method="svmRadial", metric=metric, trControl=control)

# Random Forest (rf)
set.seed(600)
fit.rf <- train(Subtypes~., data=Training_Data, method="rf", metric=metric, trControl=control)

# summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)
# compare accuracy of models
dotplot(results)
# summarize Best Model
print(fit.svm)

# estimate skill of SVM on the validation data set
predictions <- predict(fit.svm, Validation_Data)
confusionMatrix(predictions, Validation_Data$Subtypes,mode = "everything")
save.image("Machine_Learning_Project_3_-_Comparison_of_Five_Machine_Learning_Methods_in_Subtyping_Breast_Cancer_Tumors_Based_on_Gene_Expression_of_Biomarkers.RData")
