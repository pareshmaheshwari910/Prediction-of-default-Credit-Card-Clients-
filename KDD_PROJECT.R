rm(list=ls()) 
gc()

# Library 
library(ROCR)
library(caTools)
library(class)
library(dplyr)
library(ggplot2)   # For plotting
library(ggthemes)  # visualization
library(magrittr)  # Data pipelines: %>% %T>% %<>%.

library(rpart)    # Model: decision tree.
library(randomForest) # For imputing using na.roughfix()  
library(C50)    # For decision tree modeling
library(e1071) # For parameter tuning tune.nnet()
library(nnet)  # Model: neural network.
library(caret) # createDataPartition(), confusionMatrix, featurePlot()
library(pROC)  # Draw roc graph & calculate AUC
library(factoextra)


#---- Visualization-----#

rm(list=ls()) 
library('ggplot2')


# Load Dataset
Credit_Default<-read.csv("G://Stevens/Spring 2018/KDD/Paresh/Project/Credit_Card.csv")

abnormal = which(Credit_Default$EDUCATION == 0 | Credit_Default$EDUCATION == 5 | Credit_Default$EDUCATION == 6 | Credit_Default$MARRIAGE == 0)
Credit_Default = Credit_Default[-abnormal,]


# build a new data frame to use subset of the original data
credit = Credit_Default[,c("LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE")]
credit = cbind(credit,apply(Credit_Default[,7:12],1,sum))
credit = cbind(credit,apply(Credit_Default[,13:18],1,mean))
credit = cbind(credit,apply(Credit_Default[,19:24],1,mean))
credit = cbind(credit,Credit_Default[,"default.payment.next.month"])




# code some variables as factor
Credit_Default$default.payment.next.month=as.factor(Credit_Default$default.payment.next.month)
Credit_Default$SEX=as.factor(Credit_Default$SEX)
Credit_Default$EDUCATION=as.factor(Credit_Default$EDUCATION)
Credit_Default$MARRIAGE=as.factor(Credit_Default$MARRIAGE)


# change factor level names
levels(Credit_Default$SEX)[levels(Credit_Default$SEX)==1]='Male'
levels(Credit_Default$SEX)[levels(Credit_Default$SEX)==2]='Female'
levels(Credit_Default$EDUCATION)[levels(Credit_Default$EDUCATION)==1]='graduate school'
levels(Credit_Default$EDUCATION)[levels(Credit_Default$EDUCATION)==2]='university'
levels(Credit_Default$EDUCATION)[levels(Credit_Default$EDUCATION)==3]='high school'
levels(Credit_Default$EDUCATION)[levels(Credit_Default$EDUCATION)==4]='others'
levels(Credit_Default$MARRIAGE)[levels(Credit_Default$MARRIAGE)==1]='married'
levels(Credit_Default$MARRIAGE)[levels(Credit_Default$MARRIAGE)==2]='single'
levels(Credit_Default$MARRIAGE)[levels(Credit_Default$MARRIAGE)==3]='others'



# visualization
par(mar=c(3,3,3,3))
# age
qplot(AGE, data = Credit_Default, geom = "histogram", binwidth = 1, alpha = I(.5), main = "Histograms of Age")
# education
education=as.numeric(table(Credit_Default$EDUCATION))
a=plot(Credit_Default$EDUCATION,main="Education",col=c("red","blue","green","yellow"),ylim=c(0,max(education)+2000))
legend(x="topright",legend=c("graduate school","university","high school","others"),col=c("red","blue","green","yellow"),pch=15)
text(a, education+1000, sprintf("%.1f %%", 100*education/sum(education)))
# marriage
marriage=as.numeric(table(Credit_Default$MARRIAGE))
b=plot(Credit_Default$MARRIAGE,main="Marriage",col=c("cyan3","darkorchid",257),ylim=c(0,max(marriage)+2000))
legend(x="topright",legend=c("married","single","other"),col=c("cyan3","darkorchid",257),pch=15)
text(b, marriage+1000, sprintf("%.1f %%", 100*marriage/sum(marriage)))
# sex
sex=as.numeric(table(credit$SEX))
c=plot(Credit_Default$SEX,main="Gender",col=c("blue2","violetred1"),ylim=c(0,max(sex)+2000))
legend(x="topleft",legend=c("male","female"),col=c("blue2","violetred1"),pch=15)
text(c, sex+1000, sprintf("%.1f %%", 100*sex/sum(sex)))




# Q2
ggplot(Credit_Default, aes(x=SEX, y=LIMIT_BAL,col=SEX)) + geom_boxplot() + xlab("Gender") + ylab("Credit Amount") + theme(text = element_text(size=13))

ggplot(Credit_Default, aes(x=EDUCATION, y=LIMIT_BAL,col=EDUCATION)) + geom_boxplot() + xlab("Education") + ylab("Credit Amount") + theme(text = element_text(size=13))

ggplot(Credit_Default, aes(x=MARRIAGE, y=LIMIT_BAL,col=EDUCATION)) + facet_grid(.~EDUCATION) + geom_boxplot() + ylab("Credit Amount") + theme(text = element_text(size=13))



#######Decision Tree- C5.0###################
library(C50)
tree_model<-C5.0(x=training_set[,-17],y=training_set$default.payment.next.month,trials =3)
tree_model
summary(tree_model)
plot(tree_model)
test1<-predict(tree_model,test_set)
result1<-cbind(test_set$default.payment.next.month,test1)


t<-table(result1[,1],result1[,2])
sum(diag(t))/sum(t)
accuracy <- sum(test1 == test_set[,17])/nrow(test_set) * 100
accuracy
cfm <- confusionMatrix(test1, test_set$default.payment.next.month)
cfm
install.packages("pROC")
library("pROC")
pred_prob<-predict(tree_model,test_set,type='prob')
auc<-auc(test_set$default.payment.next.month,pred_prob[,2])
plot(roc(test_set$default.payment.next.month,pred_prob[,2], main="ROC curve"))
legend(.6,.2, auc, title="AUC")

###Naive bayes model##############3
install.packages('e1071', dependencies = TRUE)
library('e1071')

##Applying naive-bayes model
nbayes_dsn<- naiveBayes(default.payment.next.month ~.,data=Credit_Default) ## train the model with training dataset considering all columns
category_dsn<-predict(nbayes_dsn,Credit_Default)
category_dsn
table(Nbayes_train=category_dsn,Credit_Default$default.payment.next.month)
NB_wrong_dsn<-sum(category_dsn!=Credit_Default$default.payment.next.month)

NB_error_rate_dsn<-NB_wrong_dsn/length(category_dsn)
NB_error_rate_dsn

#Calculating accuracy for Naive Bayes model
prediction <- predict(nbayes_dsn, test_set)
?table()

t<-table(actual = test_set[,24], prediction)
accuracy <- sum(prediction == test_set$default.payment.next.month)/nrow(test_set) *100
accuracy

cfm <- confusionMatrix(prediction, test_set$default.payment.next.month)
cfm
#Plot ROC curve & find AUC

library(ROCR)
pred_test_naive<-predict(nbayes_dsn, newdata = test_set, type="raw")
p_test_naive<-prediction(pred_test_naive[,2], test_set$default.payment.next.month)
perf_naive<-performance(p_test_naive, "tpr", "fpr")
plot(perf_naive)
performance(p_test_naive, "auc")@y.values



#############KNN#####################

mmnorm <-function(x,minx,maxx) {z<-((x-minx)/(maxx-minx))
return(z)
}

for(i in 2:23){
  temp.vect <- as.vector(Credit_Default[,i])
  Credit_Default[,i] <- as.data.frame(mmnorm(temp.vect, min(temp.vect), max(temp.vect)))
}
## finding the optimal value of K using elbow method
install.packages('tidyverse')
library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
library(factoextra)

distance<-get_dist(Credit_Default)


set.seed(123)

# function to compute total within-cluster sum of square 
wss <- function(k) {
  kmeans(Credit_Default, k, nstart = 10 )$tot.withinss
}

# Compute and plot wss for k = 1 to k = 30
k.values <- 1:30

# extract wss for 2-30 clusters
wss_values <- sapply(k.values, wss)

plot(k.values, wss_values,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")



knn_fit<-knn(train=training_set[-24],test=test_set[,-24],cl=training_set[,24],k=6)

cm = as.matrix(table(Actual = test_set[,24], Prediction = knn_fit))
library('caret')
confusionMatrix(cm,positive = '0')
accuracy<-sum(diag(cm))/nrow(test_set)

##Applying random forest model
install.packages("randomForest")
library(randomForest)

## applying random forest model
rf_fit <- randomForest(default.payment.next.month~., data = training_set,importance= TRUE)
rf_fit
importance(rf_fit)
varImpPlot(rf_fit)


prediction <- predict(rf_fit, test_set)
table(actual = test_set[,24], prediction)
accuracy <- sum(prediction == test_set[,24])/nrow(test_set)
accuracy



#####------ANN---------##############333


remove(list=ls())
Credit_Default<-read.csv("C://Users/Rahul-HP/Desktop/CS_513/data/Credit_Card.csv")


Credit_Default$EDUCATION[Credit_Default$EDUCATION==0] <- NA
Credit_Default$EDUCATION[Credit_Default$EDUCATION==5] <- NA
Credit_Default$EDUCATION[Credit_Default$EDUCATION==6] <- NA
Credit_Default$MARRIAGE[Credit_Default$MARRIAGE==0] <- NA
Credit_Default<-na.omit(Credit_Default)



#Min - Max Normalization
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

Credit_Default <- as.data.frame(lapply(Credit_Default, normalize))

#Seperate the Default and Non Default Values
De<-ifelse(Credit_Default$default.payment.next.month==1,1,0)
NDe<-ifelse(Credit_Default$default.payment.next.month==0,1,0)

Credit_New<- na.omit(data.frame(Credit_Default,De,NDe))


#Splitting data into Training and Testing
set.seed(12345)
split = sample.split(Credit_New$default.payment.next.month, SplitRatio = 0.7)
training = subset(Credit_New, split == TRUE)
test = subset(Credit_New, split == FALSE)




library("neuralnet")


net_bc2  <- neuralnet(De+NDe~LIMIT_BAL+SEX+EDUCATION+MARRIAGE+AGE+PAY_0+PAY_2+PAY_3+PAY_4+PAY_5+PAY_6+BILL_AMT1+BILL_AMT2+BILL_AMT3+BILL_AMT4+BILL_AMT5+BILL_AMT6+PAY_AMT1+PAY_AMT2+PAY_AMT3+PAY_AMT4+PAY_AMT5+PAY_AMT6
                      ,training, hidden=3, threshold=0.10)

#Plot the neural network
plot(net_bc2)



net_bc2_results <- neuralnet::compute(net_bc2, test[,c(-1,-25,-26,-27)]) 
class(net_bc2_results$net.result)

str(net_bc2_results)
View(net_bc2_results)

resutls<-data.frame(Actual_Default=test$De,
                    Actual_NonDefault=test$NDe,
                    ANN_Default=round(net_bc2_results$net.result[,1]),
                    ANN_NonDefault=round(net_bc2_results$net.result[,2]))

resutls2<-data.frame(Actual_Default=test$De,
                     Actual_NonDefault=test$NDe,
                     ANN_Default=round(net_bc2_results$net.result[,1]),
                     ANN_NonDefault=round(net_bc2_results$net.result[,2])
                     ,Prediction=ifelse(round(net_bc2_results$net.result[,1])==1,'D','ND'))

cm <- table(Actual=resutls2$Actual_NonDefault,Prediction=resutls2$Prediction)
cm

sum(diag(cm))/8880
wrong<- (round(net_bc2_results$net.result[,1])!=test$De )
error_rate<-sum(wrong)/length(wrong)
error_rate


library(ROCR)
nn.pred = prediction(net_bc2_results, net_bc2_results$net.result)
pref <- performance(nn.pred, "tpr", "fpr")
plot(pref)