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
knitr::opts_chunk$set(echo = FALSE)
library(plyr)
library(dplyr)
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:plyr':
    ## 
    ##     arrange, count, desc, failwith, id, mutate, rename, summarise,
    ##     summarize

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
library(xgboost)
```

    ## Warning: package 'xgboost' was built under R version 3.5.2

    ## 
    ## Attaching package: 'xgboost'

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     slice

``` r
library(caret)
```

    ## Loading required package: lattice

    ## Loading required package: ggplot2

``` r
library(ggplot2)
```

Here is an example time series of the water temperature from July 2017.

    ## Warning: Removed 33 rows containing missing values (geom_point).

![](toshare_files/figure-markdown_github/timeseries-1.png)

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

These included: timefromlast == time since the last measurement first\_diff == the first difference of the zscore second\_diff == the second difference of the zscore, I like to think about this as acceleration (+ or -) in the measurements. previous.value == this is the zscore of the previous measurement. is.previous.identical == a logical feature for whether or not the observation is identical to the previous observation.

Balancing the data
==================

There are a lot more 'good' values than bad/flagged ones (like 97% good data). So I decided to balance the data between the good and bad for the model.

    ## Loading objects:
    ##   alldat

Base case
=========

The base case that I used here was a simple threshold - so do observed values fall outside a range of expected values for a given parameter? For example, water temperature is fresh water never below 0 deg C and won't get above 40 deg C at the latitude where this lake is located. I didn't know the actual ranges that the data owners used but I was able to reconstruct them from the good data and a some experience working with these types of sensors.

    ## Loading objects:
    ##   alldat

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

Create train, test and validation datasets
------------------------------------------

Okay, we are ready to build the model.

Build the xgboost model to predict bad data from our buoy
---------------------------------------------------------

I did not do a lot of hyperparameterization but used the xgboosts defaults as a reasonable starting point. However, I set max\_depth = 2, since I don't have a lot of features the default of 6 didn't make much sense.

    ## [1]  train-error:0.307935+0.000365   test-error:0.308067+0.003021 
    ## Multiple eval metrics are present. Will use test_error for early stopping.
    ## Will train until test_error hasn't improved in 20 rounds.
    ## 
    ## [11] train-error:0.250165+0.006043   test-error:0.250457+0.006327 
    ## [21] train-error:0.219223+0.003138   test-error:0.220430+0.004571 
    ## [31] train-error:0.200572+0.001439   test-error:0.201564+0.002634 
    ## [41] train-error:0.194499+0.001477   test-error:0.195695+0.001665 
    ## [51] train-error:0.192850+0.003055   test-error:0.193874+0.003219 
    ## [61] train-error:0.189354+0.001452   test-error:0.189897+0.002306 
    ## [71] train-error:0.185881+0.001064   test-error:0.186740+0.001220 
    ## [81] train-error:0.184063+0.000897   test-error:0.184931+0.001220 
    ## [91] train-error:0.182352+0.000599   test-error:0.183447+0.001668 
    ## [100]    train-error:0.181585+0.000720   test-error:0.182462+0.001962

``` r
watchlist <- list(train = dtrain, test = dtest)

bst <- xgb.train(data=dtrain, max_depth=2, 
                eta=0.3, nrounds=100, eval_metric = "error",
                watchlist=watchlist, objective = "binary:logistic")
```

    ## [1]  train-error:0.307563    test-error:0.307757 
    ## [2]  train-error:0.300258    test-error:0.300279 
    ## [3]  train-error:0.299195    test-error:0.299532 
    ## [4]  train-error:0.297124    test-error:0.297289 
    ## [5]  train-error:0.291777    test-error:0.291556 
    ## [6]  train-error:0.293113    test-error:0.292642 
    ## [7]  train-error:0.292715    test-error:0.292393 
    ## [8]  train-error:0.261180    test-error:0.262395 
    ## [9]  train-error:0.259768    test-error:0.261060 
    ## [10] train-error:0.245715    test-error:0.246942 
    ## [11] train-error:0.244440    test-error:0.245589 
    ## [12] train-error:0.246451    test-error:0.247868 
    ## [13] train-error:0.250433    test-error:0.251055 
    ## [14] train-error:0.246054    test-error:0.247744 
    ## [15] train-error:0.246226    test-error:0.247815 
    ## [16] train-error:0.234523    test-error:0.236759 
    ## [17] train-error:0.228548    test-error:0.228944 
    ## [18] train-error:0.219189    test-error:0.220719 
    ## [19] train-error:0.221106    test-error:0.222107 
    ## [20] train-error:0.213925    test-error:0.215111 
    ## [21] train-error:0.213445    test-error:0.214613 
    ## [22] train-error:0.213219    test-error:0.214648 
    ## [23] train-error:0.211528    test-error:0.212547 
    ## [24] train-error:0.211136    test-error:0.211426 
    ## [25] train-error:0.211000    test-error:0.211568 
    ## [26] train-error:0.209285    test-error:0.209913 
    ## [27] train-error:0.208905    test-error:0.209592 
    ## [28] train-error:0.199909    test-error:0.199694 
    ## [29] train-error:0.199707    test-error:0.199516 
    ## [30] train-error:0.199404    test-error:0.199017 
    ## [31] train-error:0.199968    test-error:0.200121 
    ## [32] train-error:0.199950    test-error:0.200068 
    ## [33] train-error:0.199974    test-error:0.199925 
    ## [34] train-error:0.199814    test-error:0.199818 
    ## [35] train-error:0.197114    test-error:0.197095 
    ## [36] train-error:0.197114    test-error:0.197077 
    ## [37] train-error:0.196490    test-error:0.196382 
    ## [38] train-error:0.196336    test-error:0.195920 
    ## [39] train-error:0.196936    test-error:0.196276 
    ## [40] train-error:0.196740    test-error:0.196062 
    ## [41] train-error:0.196680    test-error:0.196222 
    ## [42] train-error:0.196609    test-error:0.196187 
    ## [43] train-error:0.196799    test-error:0.196436 
    ## [44] train-error:0.196787    test-error:0.196436 
    ## [45] train-error:0.195452    test-error:0.195012 
    ## [46] train-error:0.195250    test-error:0.194727 
    ## [47] train-error:0.195315    test-error:0.194994 
    ## [48] train-error:0.195286    test-error:0.194122 
    ## [49] train-error:0.195250    test-error:0.194122 
    ## [50] train-error:0.195499    test-error:0.194656 
    ## [51] train-error:0.193422    test-error:0.192110 
    ## [52] train-error:0.193387    test-error:0.192092 
    ## [53] train-error:0.194075    test-error:0.192679 
    ## [54] train-error:0.194692    test-error:0.193285 
    ## [55] train-error:0.193571    test-error:0.192039 
    ## [56] train-error:0.191820    test-error:0.190454 
    ## [57] train-error:0.190230    test-error:0.189030 
    ## [58] train-error:0.190188    test-error:0.189226 
    ## [59] train-error:0.190301    test-error:0.189439 
    ## [60] train-error:0.190562    test-error:0.189635 
    ## [61] train-error:0.190562    test-error:0.189635 
    ## [62] train-error:0.190503    test-error:0.188745 
    ## [63] train-error:0.189387    test-error:0.187802 
    ## [64] train-error:0.189357    test-error:0.187748 
    ## [65] train-error:0.189316    test-error:0.188638 
    ## [66] train-error:0.187310    test-error:0.186181 
    ## [67] train-error:0.187298    test-error:0.186235 
    ## [68] train-error:0.185542    test-error:0.184757 
    ## [69] train-error:0.184764    test-error:0.184401 
    ## [70] train-error:0.184640    test-error:0.184277 
    ## [71] train-error:0.184586    test-error:0.184259 
    ## [72] train-error:0.184248    test-error:0.184010 
    ## [73] train-error:0.183963    test-error:0.183636 
    ## [74] train-error:0.183939    test-error:0.183618 
    ## [75] train-error:0.183910    test-error:0.183564 
    ## [76] train-error:0.183690    test-error:0.183315 
    ## [77] train-error:0.183518    test-error:0.183191 
    ## [78] train-error:0.183251    test-error:0.182639 
    ## [79] train-error:0.183245    test-error:0.182639 
    ## [80] train-error:0.183192    test-error:0.182514 
    ## [81] train-error:0.183637    test-error:0.182532 
    ## [82] train-error:0.183091    test-error:0.182194 
    ## [83] train-error:0.182213    test-error:0.181054 
    ## [84] train-error:0.182124    test-error:0.180876 
    ## [85] train-error:0.182124    test-error:0.180876 
    ## [86] train-error:0.182118    test-error:0.180894 
    ## [87] train-error:0.182669    test-error:0.181660 
    ## [88] train-error:0.182402    test-error:0.181446 
    ## [89] train-error:0.182290    test-error:0.181357 
    ## [90] train-error:0.182290    test-error:0.181357 
    ## [91] train-error:0.182889    test-error:0.182140 
    ## [92] train-error:0.182883    test-error:0.182122 
    ## [93] train-error:0.182681    test-error:0.181855 
    ## [94] train-error:0.182171    test-error:0.181820 
    ## [95] train-error:0.182177    test-error:0.181909 
    ## [96] train-error:0.182159    test-error:0.181873 
    ## [97] train-error:0.182159    test-error:0.181855 
    ## [98] train-error:0.182088    test-error:0.181784 
    ## [99] train-error:0.182094    test-error:0.181749 
    ## [100]    train-error:0.182082    test-error:0.181731

``` r
feature_names = names(train[, c(4:9,11)])
importance <- xgb.importance(feature_names = feature_names, model = bst)
xgb.plot.importance(importance)
```

![](toshare_files/figure-markdown_github/build%20the%20model-1.png)

The second difference came out as the most important feature. That is actually pretty cool because this feature should have picked up on quick fluctuations in the measurements. Most environmental variables don't change that rapidly, so really quick changes on the 2ish-minute scale probably mean something is acting up.

How does the model compare to the base case?
--------------------------------------------

Remember that the base case is often a simple range threshold. Does an observation fall outside of an expected range of values for a given parameter. There ranges are usually based on known physical properties (e.g., fresh water can't be below 0 deg C) or prior knowledge of the system (e.g. I know air temps are not going to get above 40 deg C at that latitude).

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction     0     1
    ##          0 22190  4193
    ##          1  6015 23773
    ##                                           
    ##                Accuracy : 0.8183          
    ##                  95% CI : (0.8151, 0.8215)
    ##     No Information Rate : 0.5021          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.6366          
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ##                                           
    ##             Sensitivity : 0.7867          
    ##             Specificity : 0.8501          
    ##          Pos Pred Value : 0.8411          
    ##          Neg Pred Value : 0.7981          
    ##               Precision : 0.8411          
    ##                  Recall : 0.7867          
    ##                      F1 : 0.8130          
    ##              Prevalence : 0.5021          
    ##          Detection Rate : 0.3950          
    ##    Detection Prevalence : 0.4697          
    ##       Balanced Accuracy : 0.8184          
    ##                                           
    ##        'Positive' Class : 0               
    ## 

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction     0     1
    ##          0 28205 26135
    ##          1     0  1831
    ##                                           
    ##                Accuracy : 0.5347          
    ##                  95% CI : (0.5306, 0.5389)
    ##     No Information Rate : 0.5021          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.0657          
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ##                                           
    ##             Sensitivity : 1.00000         
    ##             Specificity : 0.06547         
    ##          Pos Pred Value : 0.51905         
    ##          Neg Pred Value : 1.00000         
    ##               Precision : 0.51905         
    ##                  Recall : 1.00000         
    ##                      F1 : 0.68338         
    ##              Prevalence : 0.50213         
    ##          Detection Rate : 0.50213         
    ##    Detection Prevalence : 0.96740         
    ##       Balanced Accuracy : 0.53274         
    ##                                           
    ##        'Positive' Class : 0               
    ## 

So the only current automated check for bad data, a range check, flags about 53% of bad data. The model improved detection of bad data to 81%. Not too bad!

I really didnt do an hyperparameterization so I haven't used the validation data yet. Save that for another day.
