Bad Buoy, Bad Buoy, whatchya gonna do?
================
Taylor Leach
2/11/2019

Bad Buoy, Bad Buoy
==================

Environmental data is often collected using autonomous sensors on buoy or stream monitoring stations. These stations collect many different important parameters such as air temperature and relative humidity as well as aquatic variables like water temperature, dissolved oxygen and chlorophyll fluorescence (a measure of how much algae is in the water). These data are measured at time intervals from secs to hours depending on site, creating many thousands of data points a day and millions over the years. However, these buoy platforms and the sensors on them can sometimes act up, giving inaccurate measurements or breakdown and stop measuring altogether. A common challenge in using these high-frequency data is identifying sensor failure early and fix or remove bad data for later data analysis. Currently, bad data identification in this field is a primarily manual process, with only a simple automated processes (‘out of range’ threshold is the most common).

### Objective

I wanted to come up with a better method for identifying and flagging bad data. To do this, I built a model to predict bad/flagged in automated environmental monitoring stations.

### Data

The data I am using here come from an automated buoy - names 'David Buoy' run by the Center for Limnology at the University of Wisconsin, Madison. <http://blog.limnology.wisc.edu/david-buoy-ready-for-year-6-on-lake-mendota/>

These data include about 4 growing seasons of air temperature, and relative humidity, water temperature, dissolved oxygen (mg/l and percent saturation), and chlorophyll and phycocyanin fluorescence which are measures of how much algea and cyanobacteria are in the water, respectively. I did a bunch of data cleaning that I am not including here.

``` r
setwd("~/Dropbox/0_Sensor_flagged_data")
load(file = 'dat.Rdata', verbose = TRUE)
```

    ## Loading objects:
    ##   alldat

``` r
range(alldat$datetime)
```

    ## [1] "2014-05-05 00:00:00 GMT" "2017-11-13 16:06:00 GMT"

``` r
knitr::opts_chunk$set(echo = TRUE)
library(plyr)
library(dplyr)
library(xgboost)
library(caret)
library(ggplot2)
```

Here is an example time series of the water temperature from July 2017.

![](toshare_files/figure-markdown_github/timeseries-1.png)

``` r
str(alldat) # at the very beginning of the data set the Temp/DO sensor was broken, so there are some NA's initially. Don't worry there are a lot more interesting data in here. 
```

    ## 'data.frame':    7766906 obs. of  11 variables:
    ##  $ datetime             : POSIXct, format: "2014-05-05 00:00:00" "2014-05-05 00:01:00" ...
    ##  $ variable             : chr  "doptotemp" "doptotemp" "doptotemp" "doptotemp" ...
    ##  $ value                : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ zscore               : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ timefromlast         : num  NA 1 1 1 1 1 1 1 1 1 ...
    ##  $ first_diff           : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ second_diff          : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ previous.value       : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ is.previous.identical: num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ flag                 : chr  "C" "C" "C" "C" ...
    ##  $ flagYN               : num  1 1 1 1 1 1 1 1 1 1 ...

I normalized all of the measured values as z-scores --&gt; (x - mu)/sd and then build several features that I thought might help predict whether or not you had bad data. Since this is time series data, I decided to only build features that rely on previous values since often times we can access real-time streaming data from these buoy platforms.

These included:

-   zscore == (x - mu)/sd (all the below features were calculated with the zscore, not the value because the values vary pretty widely between sensors.)
-   timefromlast == time since the last measurement
-   first\_diff == the first difference of the zscore
-   second\_diff == the second difference of the zscore, I like to think about this as acceleration (+ or -) in the measurements.
-   previous.value == this is the zscore of the previous measurement.
-   is.previous.identical == a logical feature for whether or not the observation is identical to the previous observation.

Balancing the data
==================

There are a lot more 'good' values than bad/flagged ones (like 97% good data). So I decided to balance the data between the good and bad for the model.

``` r
load(file = 'dat.Rdata', verbose = TRUE)
```

    ## Loading objects:
    ##   alldat

``` r
alldat = alldat[!is.na(alldat$zscore), ] # if there is no zscore we can't do much here so get rid of them
edflags = subset(alldat, flag %in%c("E", "D")) # the flags that aren't NA's or out of range (i.e. the basecase already applied)
goodonly = subset(alldat, flag =="")
alldat.small = dplyr::sample_n(goodonly, nrow(edflags)) # randomly sample so our dataset is 50:50 good and bab
alldata.balanced = rbind(edflags, alldat.small)
```

Base case
=========

The base case that I used here was a simple threshold - so do observed values fall outside a range of expected values for a given parameter? For example, water temperature is fresh water never below 0 deg C and won't get above 40 deg C at the latitude where this lake is located. I didn't know the actual ranges that the data owners used but I was able to reconstruct them from the good data and a some experience working with these types of sensors.

``` r
detach(package:plyr)
load(file = 'dat.Rdata', verbose = TRUE)
```

    ## Loading objects:
    ##   alldat

``` r
inrange = subset(alldat, flag == "")
gg = inrange %>%
  group_by(variable) %>%
  summarize(low = min(value, na.rm = T), high = max(value, na.rm = T))
gg
```

    ## # A tibble: 7 x 3
    ##   variable       low    high
    ##   <chr>        <dbl>   <dbl>
    ## 1 airTL        -9.67    39.4
    ## 2 chlor.x   -2784.   49278. 
    ## 3 doptoppm      0       23.6
    ## 4 doptosat      0      205. 
    ## 5 doptotemp     3.39    29  
    ## 6 phyco     -9350.   49272. 
    ## 7 rhL           0      102.

I rounded a bit and then applied these thresholds to all of the data. This had to be done pretty manually because the thresholds were unique for each variable.

``` r
wtemp = subset(alldata.balanced , variable == "doptotemp")
wtemp$flagBASE = c(rep(0, length(wtemp$value)))
wtemp$flagBASE = as.numeric(is.na(wtemp$value) | (wtemp$value > 40 | wtemp$value < 0))

airtemp = subset(alldata.balanced , variable =="airTL")
airtemp$flagBASE = c(rep(0, length(airtemp$value)))
airtemp$flagBASE = as.numeric(is.na(airtemp$value) | (airtemp$value > 40 | airtemp$value < -10))

doobs = subset(alldata.balanced , variable =="doptoppm")
doobs$flagBASE = c(rep(0, length(doobs$value)))
doobs$flagBASE = as.numeric(is.na(doobs$value) | (doobs$value > 24 | doobs$value < 0))

doobspercent = subset(alldata.balanced , variable == "doptosat")
doobspercent$flagBASE = c(rep(0, length(doobspercent$value)))
doobspercent$flagBASE = as.numeric(is.na(doobspercent$value) | (doobspercent$value > 205 | doobspercent$value < 0))

phytos = subset(alldata.balanced , variable %in% c("chlor.x", "phyco"))
phytos$flagBASE = c(rep(0, length(phytos$value)))
phytos$flagBASE = as.numeric(is.na(phytos$value) | (phytos$value > 50000 | phytos$value < -3000))

relH = subset(alldata.balanced , variable =="rhL")
relH$flagBASE = c(rep(0, length(relH$value)))
relH$flagBASE = as.numeric((is.na(relH$value) | (relH$value <= 0 | relH$value > 102)))

alldat.balanced.base = rbind(wtemp, airtemp,doobs,doobspercent, phytos,relH)
```

Create train, test and validation datasets
------------------------------------------

``` r
spec = c(train = .6, test = .2, validate = .2)
g = sample(cut(
  seq(nrow(alldat.balanced.base)), 
  nrow(alldat.balanced.base)*cumsum(c(0,spec)),
  labels = names(spec)
))

res = split(alldat.balanced.base, g)
train = res[["train"]]
test = res[["test"]]
validation = res[["validate"]]
```

Okay, we are ready to build the model.

Build the xgboost model to predict bad data from our buoy
---------------------------------------------------------

``` r
# create xbg.DMatrix objects for input
ttrain <- data.matrix(train[, c(4:9,11)]) 
ttest  <- data.matrix(test[, c(4:9,11)])
dtrain <- xgb.DMatrix(ttrain[,c(1:6)], label=ttrain[,7])
dtest  <- xgb.DMatrix(ttest[,c(1:6)], label=ttest[,7])
```

I did not do a lot of hyperparameterization but used the xgboosts defaults as a reasonable starting point. However, I set max\_depth = 2, since I don't have a lot of features the default of 6 didn't make much sense.

``` r
# How many iterations?
params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.3, gamma=0, max_depth=2, min_child_weight=1, subsample=1, colsample_bytree=1)
xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 100, nfold = 5, showsd = T, stratified = T, print_every_n = 10, early_stopping_rounds = 20, maximize = F)
```

    ## [1]  train-error:0.308712+0.000656   test-error:0.308809+0.002193 
    ## Multiple eval metrics are present. Will use test_error for early stopping.
    ## Will train until test_error hasn't improved in 20 rounds.
    ## 
    ## [11] train-error:0.245226+0.001326   test-error:0.245781+0.002962 
    ## [21] train-error:0.218570+0.006320   test-error:0.219818+0.005945 
    ## [31] train-error:0.205284+0.000897   test-error:0.206389+0.003538 
    ## [41] train-error:0.196569+0.000653   test-error:0.197909+0.001311 
    ## [51] train-error:0.191112+0.002727   test-error:0.191921+0.003408 
    ## [61] train-error:0.187863+0.001037   test-error:0.189055+0.001588 
    ## [71] train-error:0.185786+0.000799   test-error:0.186539+0.001989 
    ## [81] train-error:0.183981+0.000724   test-error:0.184740+0.001511 
    ## [91] train-error:0.182221+0.000575   test-error:0.182996+0.001637 
    ## [100]    train-error:0.180840+0.000806   test-error:0.181293+0.001232

``` r
# best iteration was at 100 - this could probably use some more exploration.
```

``` r
watchlist <- list(train = dtrain, test = dtest)

bst <- xgb.train(data=dtrain, max_depth=2, 
                eta=0.3, nrounds=100, eval_metric = "error",
                watchlist=watchlist, objective = "binary:logistic")
```

    ## [1]  train-error:0.308423    test-error:0.308220 
    ## [2]  train-error:0.301213    test-error:0.301704 
    ## [3]  train-error:0.299617    test-error:0.300244 
    ## [4]  train-error:0.295617    test-error:0.294956 
    ## [5]  train-error:0.292798    test-error:0.292749 
    ## [6]  train-error:0.293101    test-error:0.292731 
    ## [7]  train-error:0.291766    test-error:0.291254 
    ## [8]  train-error:0.258806    test-error:0.258158 
    ## [9]  train-error:0.258842    test-error:0.258318 
    ## [10] train-error:0.259127    test-error:0.258461 
    ## [11] train-error:0.247181    test-error:0.246764 
    ## [12] train-error:0.249585    test-error:0.250200 
    ## [13] train-error:0.245609    test-error:0.245874 
    ## [14] train-error:0.246208    test-error:0.246390 
    ## [15] train-error:0.239306    test-error:0.239786 
    ## [16] train-error:0.239016    test-error:0.239554 
    ## [17] train-error:0.239217    test-error:0.239910 
    ## [18] train-error:0.227195    test-error:0.227769 
    ## [19] train-error:0.224690    test-error:0.225276 
    ## [20] train-error:0.224150    test-error:0.224386 
    ## [21] train-error:0.217433    test-error:0.217319 
    ## [22] train-error:0.214614    test-error:0.214737 
    ## [23] train-error:0.214323    test-error:0.213936 
    ## [24] train-error:0.208834    test-error:0.209806 
    ## [25] train-error:0.208246    test-error:0.209165 
    ## [26] train-error:0.208068    test-error:0.209005 
    ## [27] train-error:0.207789    test-error:0.208649 
    ## [28] train-error:0.207677    test-error:0.208506 
    ## [29] train-error:0.207315    test-error:0.208310 
    ## [30] train-error:0.206852    test-error:0.207118 
    ## [31] train-error:0.206769    test-error:0.206708 
    ## [32] train-error:0.204270    test-error:0.205462 
    ## [33] train-error:0.204146    test-error:0.205284 
    ## [34] train-error:0.203819    test-error:0.204928 
    ## [35] train-error:0.203677    test-error:0.204732 
    ## [36] train-error:0.198306    test-error:0.198679 
    ## [37] train-error:0.198217    test-error:0.198537 
    ## [38] train-error:0.198087    test-error:0.198430 
    ## [39] train-error:0.197576    test-error:0.197664 
    ## [40] train-error:0.197066    test-error:0.197255 
    ## [41] train-error:0.196995    test-error:0.196952 
    ## [42] train-error:0.196099    test-error:0.196062 
    ## [43] train-error:0.196010    test-error:0.195884 
    ## [44] train-error:0.193517    test-error:0.193445 
    ## [45] train-error:0.192651    test-error:0.192573 
    ## [46] train-error:0.192793    test-error:0.192715 
    ## [47] train-error:0.192782    test-error:0.192751 
    ## [48] train-error:0.191939    test-error:0.191736 
    ## [49] train-error:0.191535    test-error:0.191416 
    ## [50] train-error:0.189844    test-error:0.189493 
    ## [51] train-error:0.189245    test-error:0.188709 
    ## [52] train-error:0.189666    test-error:0.189742 
    ## [53] train-error:0.189001    test-error:0.188389 
    ## [54] train-error:0.188918    test-error:0.188300 
    ## [55] train-error:0.188473    test-error:0.188175 
    ## [56] train-error:0.187998    test-error:0.187766 
    ## [57] train-error:0.187915    test-error:0.187143 
    ## [58] train-error:0.187168    test-error:0.186164 
    ## [59] train-error:0.187340    test-error:0.186306 
    ## [60] train-error:0.188177    test-error:0.186840 
    ## [61] train-error:0.188177    test-error:0.186840 
    ## [62] train-error:0.187690    test-error:0.186449 
    ## [63] train-error:0.187464    test-error:0.186306 
    ## [64] train-error:0.187464    test-error:0.186306 
    ## [65] train-error:0.188260    test-error:0.186644 
    ## [66] train-error:0.186954    test-error:0.185612 
    ## [67] train-error:0.187334    test-error:0.186253 
    ## [68] train-error:0.186070    test-error:0.185131 
    ## [69] train-error:0.185583    test-error:0.184704 
    ## [70] train-error:0.184770    test-error:0.184668 
    ## [71] train-error:0.184598    test-error:0.184437 
    ## [72] train-error:0.184675    test-error:0.184312 
    ## [73] train-error:0.184723    test-error:0.184330 
    ## [74] train-error:0.184604    test-error:0.184152 
    ## [75] train-error:0.184770    test-error:0.184045 
    ## [76] train-error:0.184758    test-error:0.184045 
    ## [77] train-error:0.185530    test-error:0.184401 
    ## [78] train-error:0.184284    test-error:0.183102 
    ## [79] train-error:0.184236    test-error:0.183689 
    ## [80] train-error:0.184201    test-error:0.183653 
    ## [81] train-error:0.183969    test-error:0.183458 
    ## [82] train-error:0.184236    test-error:0.183707 
    ## [83] train-error:0.184028    test-error:0.183066 
    ## [84] train-error:0.183625    test-error:0.182959 
    ## [85] train-error:0.183625    test-error:0.182888 
    ## [86] train-error:0.183833    test-error:0.183155 
    ## [87] train-error:0.183465    test-error:0.182870 
    ## [88] train-error:0.182557    test-error:0.182194 
    ## [89] train-error:0.182693    test-error:0.182336 
    ## [90] train-error:0.182681    test-error:0.182336 
    ## [91] train-error:0.182675    test-error:0.182354 
    ## [92] train-error:0.182705    test-error:0.182318 
    ## [93] train-error:0.182414    test-error:0.182122 
    ## [94] train-error:0.181827    test-error:0.181535 
    ## [95] train-error:0.181352    test-error:0.181036 
    ## [96] train-error:0.181370    test-error:0.180876 
    ## [97] train-error:0.181233    test-error:0.180502 
    ## [98] train-error:0.181204    test-error:0.180502 
    ## [99] train-error:0.181204    test-error:0.180502 
    ## [100]    train-error:0.181043    test-error:0.180378

``` r
feature_names = names(train[, c(4:9,11)])
importance <- xgb.importance(feature_names = feature_names, model = bst)
xgb.plot.importance(importance)
```

![](toshare_files/figure-markdown_github/build%20the%20model-1.png)

The second difference came out as the most important feature. That is actually pretty cool because this feature should have picked up on quick and large fluctuations in the measurements. Most environmental variables don't change that rapidly, so really quick changes on the 2ish-minute scale probably mean something is acting up.

How does the model compare to the base case?
--------------------------------------------

Remember that the base case is often a simple range threshold - does an observation fall outside of an expected range of values for a given parameter? These ranges are usually based on known physical properties (e.g., fresh water can't be below 0 deg C) or prior knowledge of the system (e.g. I know air temps are not going to get above 40 deg C at that latitude).

``` r
pred <- predict(bst, dtest)
prediction <- ifelse (pred > 0.5,1,0) # turn the probabilities into 1 and 0 (1 = bad data)

# Model
caret::confusionMatrix(as.factor(prediction), as.factor(ttest[,7]), mode = "everything") 
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction     0     1
    ##          0 22258  4264
    ##          1  5868 23781
    ##                                           
    ##                Accuracy : 0.8196          
    ##                  95% CI : (0.8164, 0.8228)
    ##     No Information Rate : 0.5007          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.6393          
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ##                                           
    ##             Sensitivity : 0.7914          
    ##             Specificity : 0.8480          
    ##          Pos Pred Value : 0.8392          
    ##          Neg Pred Value : 0.8021          
    ##               Precision : 0.8392          
    ##                  Recall : 0.7914          
    ##                      F1 : 0.8146          
    ##              Prevalence : 0.5007          
    ##          Detection Rate : 0.3963          
    ##    Detection Prevalence : 0.4722          
    ##       Balanced Accuracy : 0.8197          
    ##                                           
    ##        'Positive' Class : 0               
    ## 

``` r
# model accuracy is 81%

# Basecase
caret::confusionMatrix(as.factor(test$flagBASE), as.factor(ttest[,7]), mode = "everything") 
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction     0     1
    ##          0 28126 26215
    ##          1     0  1830
    ##                                           
    ##                Accuracy : 0.5333          
    ##                  95% CI : (0.5292, 0.5374)
    ##     No Information Rate : 0.5007          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.0653          
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ##                                           
    ##             Sensitivity : 1.00000         
    ##             Specificity : 0.06525         
    ##          Pos Pred Value : 0.51758         
    ##          Neg Pred Value : 1.00000         
    ##               Precision : 0.51758         
    ##                  Recall : 1.00000         
    ##                      F1 : 0.68212         
    ##              Prevalence : 0.50072         
    ##          Detection Rate : 0.50072         
    ##    Detection Prevalence : 0.96742         
    ##       Balanced Accuracy : 0.53263         
    ##                                           
    ##        'Positive' Class : 0               
    ## 

``` r
#basecase accuracy %53
```

So the only current automated check for bad data, a range check, flags about 53% of bad data. The model improved detection of bad data to 81%. Not too bad!

I really didn't do an hyperparameterization so I haven't used the validation data yet. Save that for another day.
