John Hopkins Practical Machine Learning Report
================

Introduction
------------

The project for this course is to use machine learning to predict the manner in which a group did excercise. The following objectives are to be met as part of this report: \* Describe how you built your model \* How you used cross validation \* Why you made the choices you did

### Project Setup

First, include required libraries, create folders, and download files

``` r
# Include libraries
library(caret)
```

``` r
# Create data folder if it does not exist
dataFolder = "data"
if (!file.exists(dataFolder)) {
        dir.create(dataFolder)
}
```

``` r
# Download test files if they do not exist
trainInput <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testInput <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
trainDisk <- file.path(dataFolder,"training.csv")
testDisk <- file.path(dataFolder, "test.csv")

if (!file.exists(trainDisk)) {
        download.file(trainInput, destfile = trainDisk, method = "curl")
}

if (!file.exists(testDisk)) {
        download.file(testInput, destfile = testDisk, method = "curl")
}
```

### Prepare Data

Next we need to load, clean, and partition our data sets

``` r
# Read the downloaded csv files into data frames
trainCsv <- read.csv(trainDisk)
testCsv <- read.csv(trainDisk)

# Next I used View(trainCsv) and I see lots of NA Values. Lets remove those first.
trainCsv <- trainCsv[, colSums(is.na(trainCsv)) == 0]
testCsv <-  testCsv[, colSums(is.na(testCsv)) == 0]

# Again I used View(trainCsv) and it appears columns 3-8 (timestamp / window) are not useful
dropCol <- c("raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "new_window", "num_window")
trainCsv <- trainCsv[, -which(names(trainCsv) %in% dropCol)]
testCsv <- testCsv[, -which(names(testCsv) %in% dropCol)]
```

GitHub Documents
----------------

This is an R Markdown format used for publishing markdown documents to GitHub. When you click the **Knit** button all R code chunks are run and a markdown file (.md) suitable for publishing to GitHub is generated.

Including Code
--------------

You can include R code in the document as follows:

``` r
summary(cars)
```

    ##      speed           dist       
    ##  Min.   : 4.0   Min.   :  2.00  
    ##  1st Qu.:12.0   1st Qu.: 26.00  
    ##  Median :15.0   Median : 36.00  
    ##  Mean   :15.4   Mean   : 42.98  
    ##  3rd Qu.:19.0   3rd Qu.: 56.00  
    ##  Max.   :25.0   Max.   :120.00

Including Plots
---------------

You can also embed plots, for example:

![](report_files/figure-markdown_github-ascii_identifiers/pressure-1.png)

Note that the `echo = FALSE` parameter was added to the code chunk to prevent printing of the R code that generated the plot.
