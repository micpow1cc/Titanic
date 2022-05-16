library(MASS)
library(reshape)
library(caret)
library(randomForest)
library(e1071)
library(class)

#wczytanie zbioru danych
df2 <- read.csv(file.choose())
# zmiany konkretnych zmiennych na factor, by byly uzyteczne w predykcji
df2$Sex<- as.factor(df2$Sex)
df2$Pclass<- as.factor(df2$Pclass)
df2$Parch<- as.factor((df2$Parch))
df2$Survived<- as.factor(df2$Survived)
df2$Title<- as.factor(df2$Title)
df2$Embarked<-as.factor(df2$Embarked) 
df2$SibSp<-as.factor(df2$SibSp) 

#Survival - Survival (0 = No; 1 = Yes). Not included in test.csv file.
#Pclass - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
#Name - Name
#Sex - Sex
#Age - Age
#Sibsp - Number of Siblings/Spouses Aboard
#Parch - Number of Parents/Children Aboard
#Ticket - Ticket Number
#Fare - Passenger Fare
#Cabin - Cabin
#Embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

# funkcjaznajdujaca brakujace wartosci w dataframe df2, po ich znalezieniu mozna przystapic do przygotowywania danych do predykcji
df2.miss <- melt(apply(df2[, -2], 2, function(x) sum(is.na(x) | x=="NA")))
cbind(row.names(df2.miss)[df2.miss$value>0],df2.miss[df2.miss$value>0,])
df2 <- subset (df2, select = -Cabin)


table(df2$Embarked)
# funkcja uzupelniajaca brakujace wartosci Embarked na podstawie najczesciej wystepujacej wartosci w zmiennej Embarked
df2$Embarked[which(is.na(df2$Embarked)| df2$Embarked=="")] <- 'S'

df2$Fare[which(is.na(df2$Fare))] <- 0

# funkcja uzupelniajaca brakujace wartosci Pclass na podstawie mediany (najczejsciej wystepujacej wartosci klasy) i dodajaca kolumne price w ktorej 
# wartosciami jest cena bedaca wyliczona na podstawie parametru Pclass
pclass.price<-aggregate(df2$Price, by = list(df2$Pclass), FUN = function(x) median(x, na.rm = T))
df2[which(df2$Price==0), "Price"] <- apply(df2[which(df2$Price==0), ] , 1, function(x) pclass.price[pclass.price[, 1]==x["Pclass"], 2])
# funkcja znajdujaca te same bilety i dzielaca oplate za bilet na pasazera.
ticket.count <- aggregate(df2$Ticket, by=list(df2$Ticket), function(x) sum(!is.na(x)))
df2$Price <- apply(df2,1,function(x) as.numeric(x["Fare"])/ticket.count[which(ticket.count[,1]==x["Ticket"]),2])
# funkcja "wyciagajaca" tytul ze zmiennej Name
df2$Title<-regmatches(as.character(df2$Name),regexpr("\\,[A-z ]{1,20}\\.", as.character(df2$Name)))
df2$Title<-unlist(lapply(df2$Title,FUN=function(x) substr(x, 3, nchar(x)-1)))
table(df2$Title)
# funkcje zmniejszajace liczbe tytulow do kilku.
df2$Title[which(df2$Title %in% c("Mme", "Mlle"))] <- "Miss"
df2$Title[which(df2$Title %in% c("Lady", "Ms", "the Countess", "Dona"))] <- "Mrs"
df2$Title[which(df2$Title %in% c("Dr", "Master") & df2$Sex=="female")] <- "Mrs"
df2$Title[which(df2$Title  %in% c("Dr", "Master") & df2$Sex=="male")] <- "Mr"

df2$Title[which(df2$Title=="Master" & df2$Sex=="male")] <- "Mr"
df2$Title[which(df2$Title %in% c("Capt", "Col", "Don", "Jonkheer", "Major", "Rev", "Sir"))] <- "Mr"
df2$Title<-as.factor(df2$Title) 
df2$Title<- as.character(df2$Title)
# funkcja wilczajaca wiek pasazera na podstawie jego tytulu
title.age<-aggregate(df2$Age,by = list(df2$Title), FUN = function(x) median(x, na.rm = T))
df2[is.na(df2$Age), "Age"] <- apply(df2[is.na(df2$Age), ] , 1, function(x) title.age[title.age[, 1]==x["Title"], 2])


ind <- sample(2,nrow(df2), replace=TRUE, prob = c(0.7,0.3))

train <- df2[ind==1,]
test <- df2[ind==2,]
#############################################################
# RANDOM FOREST 
#############################################################
rf <- randomForest(Survived ~ .,data=train,
                   ntree=300,
                   mtry=3,
                   importance=TRUE,
                   proximity=TRUE)
p1 <- predict(rf, train)
p2 <- predict(rf,test)
confusionMatrix(p2,test$Survived)
print(rf)
varImpPlot(rf)
plot(rf)
#random forest accuracy =0.8051 z proby testowej
# strojenie modelu
t <-tuneRF(train[,-13],train[,13],
       stepFactor = 0.5,
       plot=TRUE,
       ntreeTry =300,
       improve=0.05)

# po strojeniu dokladnosc wynosi  ACC=0.8062

####################################################################
# Naive Bayes
####################################################################


nb <- naiveBayes(Survived~., data=train,)
p3 <- predict(nb,test)
confMatrix <- table(p3,test$Survived)
confusionMatrix(confMatrix)
print(nb)

# Naive Bayes klasyfikator ACC= 0.7752


####################################################################
# k nearest neighbors
####################################################################
knn_reduced <- data.frame(knndataset$df2.Survived,knndataset$df2.Sex,knndataset$df2.Age,
                          knndataset$df2.Fare,knndataset$df2.Price)

knndataset <- data.frame(df2$Survived,df2$Pclass,df2$Sex,df2$Age,
                         df2$SibSp,df2$Parch,df2$Fare,df2$Embarked,
                         df2$Price,df2$Title)


normalize <- function(x) {
  return ((x-min(x)/(max(x)-min(x))))
}
knn_norm <- as.data.frame(lapply(knn_reduced[,c(3,4,5)],normalize))

ind <- sample(2,nrow(knn_reduced),replace = TRUE,c(0.7,0.3))
train_knn <- knn_norm[ind,]
test_knn <- knn_norm[-ind,]
knn_target <- knn_reduced[ind,1]
knn_test_cat <- knn_reduced[-ind,1]

knn <- knn(train_knn,test_knn,cl=knn_target,k=13)

tab <- table(knn,knn_test_cat)

acc <- function(x) {sum(diag(x)/sum(rowSums(x)))*100}
acc(tab)
# KNN:  ACC=0.6760 , wartosc ta jest mniejsza od wczesniejszych klasyfikatorow
# ze wzgledu na to ze metoda knn nie moze uzywac zmiennych factor(mierzenie
# odleglosci jest nie mozliwe)


