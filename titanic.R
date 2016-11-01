library(caret)

trainDf <- read.csv('~/Downloads/train.csv')
l <- length(trainDf$Name)

#intro exploration of data
#remove(g)
#g <- ggplot(data = trainDf, 
#            aes(x = Age, color = as.factor(Survived)))
#g <- g + geom_histogram(fill="white",
#                        alpha = .5, 
#                        position="identity")
#g + facet_wrap(~Embarked + Pclass) + theme_bw()
#####



#cabin only has ~200 filled entries out of ~900
#so I'm just going to remove it completely  

trainDf <- trainDf[-which(colnames(trainDf) == "Cabin")] #remove cabin 
trainDf <- trainDf[-which(colnames(trainDf) == "Ticket")]  #remove passenger Id
trainDf <- trainDf[-which(colnames(trainDf) == "PassengerId")]	#remove ticket 
trainDf <- trainDf[-which(colnames(trainDf) == "Name")] 


#find columns with missing data
#sample to fill in data (just simple sampling based on all other data entries)
#want to do move advanced gibbs type sampling later on if needed

#haveNa   <- sapply(testDf, function(x) any(is.na(x)))		#age
#haveMiss <- sapply(trainDf, function(x) any(x == ''))		#cabin/embarked


uncompInd <- which(!complete.cases(trainDf))

for (i in 1:length(uncompInd)){
	colInd <- which(is.na(trainDf[uncompInd[i],]))
	
	while(is.na(trainDf[uncompInd[i],colInd])){
		trainDf[uncompInd[i],colInd] <- sample(trainDf[,colInd],1)
	}
}

missInd <- which(trainDf$Embarked == '')

for (i in 1:length(missInd)){
	colInd <- which(is.na(trainDf[missInd[i],]))
	
	while(trainDf[missInd[i],colInd] == ''){
		trainDf[missInd[i],colInd] <- sample(trainDf[,colInd],1)
	}
}


#change from number to factor 
trainDf$Survived = as.factor(trainDf$Survived)
trainDf$Pclass   = as.factor(trainDf$Pclass)

trainingSet <- createDataPartition(trainDf$Survived,
                                   p=0.85, list=FALSE)
trainDf     <- trainDf[trainingSet,]
validDf     <- trainDf[-trainingSet,]

mod_rf <- train(Survived ~., data = trainDf,
                method = 'rf')
pred <- predict(mod_rf, newdata = validDf)

print(confusionMatrix(pred,validDf$Survived))

#Do same on Test set

testDf <- read.csv('~/Downloads/test.csv')
idNums <- testDf$PassengerId

testDf <- testDf[-which(colnames(testDf) == "Cabin")] #remove cabin 
testDf <- testDf[-which(colnames(testDf) == "Ticket")]  #remove passenger Id
testDf <- testDf[-which(colnames(testDf) == "PassengerId")]	#remove ticket
testDf <- testDf[-which(colnames(testDf) == "Name")] 


uncompInd <- which(!complete.cases(testDf))

for (i in 1:length(uncompInd)){
	colInd <- which(is.na(testDf[uncompInd[i],]))
	
	while(is.na(testDf[uncompInd[i],colInd])){
		testDf[uncompInd[i],colInd] <- sample(testDf[,colInd],1)
	}
}

missInd <- which(testDf$Embarked == '')

for (i in 1:length(missInd)){
	colInd <- which(is.na(testDf[missInd[i],]))
	
	while(testDf[missInd[i],colInd] == ''){
		testDf[missInd[i],colInd] <- sample(testDf[,colInd],1)
	}
}

testDf$Pclass   = as.factor(testDf$Pclass)
predOnTest <- predict(mod_rf, newdata = testDf)


outputDf <- data.frame("PassengerId" = idNums, "Survived" = predOnTest)

write.csv(outputDf, file = "titanicPrediction.csv", row.names=FALSE, na="")





