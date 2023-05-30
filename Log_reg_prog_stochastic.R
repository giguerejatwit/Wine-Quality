setwd("/Users/jakegiguere/Dev/ML/Wine_Quality")
library(tidyverse)
library(ggpubr)
library(rstatix)
datatr <- read.csv(file = "winequality_red.csv", head = TRUE, sep = ",")
#data_plant_test <- read.csv(file = "Plant_data_test.csv", head = TRUE, sep = ",")
idxs <- sample(1:nrow(datatr),as.integer(0.7*nrow(datatr)))
trainx <- datatr[idxs,]
testx <- datatr[-idxs,]
ncl <- ncol(trainx)
nr <- nrow(trainx)
nclt <- ncol(testx)
nrt <- nrow(testx)

for (i in 1:(ncl-1)){
  trainx[,i] <- (trainx[,i]-mean(as.vector(trainx[,i])))/sd(as.vector(trainx[,i]))
}
for (i in 1:(nclt-1)){
  testx[,i] <- (testx[,i]-mean(as.vector(trainx[,i])))/sd(as.vector(testx[,i]))
}
trainx[,ncl] <- trainx[,ncl]/10
# print(trainx)
testx[,nclt] <- testx[,nclt]/10
# print(testx)
library(aod)
#the previous line is the library to use the following program
mylogit <- glm(trainx[,12]~., data = trainx, family = binomial(link="logit"))
zwss <- summary(mylogit)
# print(ss)
x0 <- numeric()
for (i in 1:nr){
x0[i]=1
}
x0t <- numeric()
for (i in 1:nrt){
  x0t[i]=1
}
trainx <- as.data.frame(cbind(x0,trainx))
testx <- as.data.frame(cbind(x0t,testx))
#print(trainx)
#print(testx)
#linear function
linear <- function(x,theta){
  prod <- sum(x*theta)
  return(prod)
}
#logistic function
logfun <- function(a){
  funct <- 1.0/(1.0+exp(-a))
  return(funct)
}

d <- numeric()
#these are the parameters
dimm <- ncol(trainx)-1
theta <- matrix(0,nrow=1,ncol=dimm) #this will be the old theta's
theta1 <- matrix(0,nrow=1,ncol=dimm) #this will be the new theta's
theta <- matrix(rnorm(dimm),nrow=1) #random initial guesses 
#for parameter matrix
#correction <- matrix(0,nrow=1,ncol=dimm)
alpha=0.1
theta1 <- theta
x <- trainx[,-13]
print(trainx[,-13])
nn2 <- integer()
for (i in 1:100000){
  
  ## this loops runs over all of the data
  for (j in 1:nr) {
    
    #k=sample(nr,1,replace=F)
    k=j
    xx <- as.numeric(x[k,])
    yy <- as.numeric(trainx[k,13])
    a <- linear(xx,theta)
    hh2 <- logfun(a)
    const <- yy - hh2
    theta <- theta + alpha*const*xx
  }
  nn2[i] <- i   
  #correction <- matrix(0,nrow=1,ncol=dimm)
  d=abs(theta1-theta)
  err <- max(d)
  #if difference is less that 1*10-6 exit loop with break command
  #since it has converged to within 1*10^-6
  if (err <= 1*10^-5){
    break
  }else{
    print(err) 
  }
  theta1 <- theta
}

#save parameters in data frame
datatheta <- data.frame(theta)
# print(datatheta)
