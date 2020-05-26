# Song classification using echonest data: Project Overview
* The goal of the project was to classify the genre of any song by it's given characteristics.
* The project was initialized by processing of Dataset and also some exploration related to
pairwise relationships.
* Then for easier processing the data was normalized followed by Principal Component
Analysis (PCA).
* Decison Tree and Logistic Regression Model Training
* Balancing Data
* Re-training and comaprison of results.

## Code and Resources Used 
**Python Version:** 3.4  
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn

## Model Building 

Firstly, I split the data into training and testing set in the ration of 70:30. 

I tried to two different models and evaluated them with the help of confusion matrix as to check various aspects of models. 

I tried two different models:

**Decison Tree**

**Logistic Regression**

The results revealed that the data set is unbalanced, hence I balanced the data and re-trained both models.

## Model performance
The Logistic Regression model outperformed the other approach on the test sets. 

**Decison Tree** 

*Before Balancing :*

>[[ 173   97]

>[  86 1085]]

*After Balancing :* 

[[ 216   51]

[  60 219]]     
                    
**Logistic Regression** 

*Before Balancing :* 

>[[ 147  123]

>[  35 1136]]

*After Balancing :* 

>[[ 229  38]

>[  41 238]]                         

**The Cross-Validation results for both models:**

>Decison Tree: 0.760989010989011

>Logistic Model: 0.8186813186813187
