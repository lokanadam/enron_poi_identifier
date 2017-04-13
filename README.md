# enron_poi_identifier
Machine Learning: POI identifier in enron fraud in python (sklearn)


### control variables

**testall = False (default)** - set this to **True** if all the grid test should be run 
                            and displayed. 
                            

**dataprocess = False(default)** - set this to **True** if the entire email corpus has to 
                               process from scratch and new .pkl files has to be created.
                               

**zipped = False(default)** - set this to **True** if using zipped file. 

**least_component = True(default)** - set this to **False** if the PCA should not be run in 
                                  email corpus.
                                  

**MAIN-FILE - poi_id.py**

Please run above file with the other .py files in the same folder. 

Process - 

1. The email and financial data were transformed into features, cross validated to feed the algorithms used.

2. The email corpus was processed using Nature Language Processing and fed in to SVM to improve the accuracy of the above process.

### Problems encountered and solutions:

**Small Dataset** - The initial email and financial data was small with just 140 data points

***Solution*** - Multi fold cross-validation was used to handle the problem of small dataset.

**Imbalanced Dataset** - The POI and Non-POI datapoints were not balanced in the dataset.

***Solution*** - The dataset was balanced using undersampling and oversampling the dataset.

**Too many features** - Features created using the email corpus were too many and was very process intensive

***Solution*** - PCA was used to project components in to a more manageable dimension.

### Metrics Summary

**Email and Financial dataset & SVM**

Precision: 0.72720	
Recall: 0.81083	
F-Score: 0.76675	

**Email corpus data & SVM**

Precision: 0.982078853047
Recall: 0.984431137725
F-Score 0.983253588517
