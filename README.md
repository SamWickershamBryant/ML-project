# Project Assignments 

* [Project Proposal](https://canvas.txstate.edu/courses/2179554/assignments/31569710) 
* [Project Progress](https://canvas.txstate.edu/courses/2179554/assignments/31569830)
* [Project Submission](https://canvas.txstate.edu/courses/2179554/assignments/31569848)

 **Submission Format** 
 
1. Python code with markdown documentation, images saved in .jpg or .png format, and README.md as a project report OR
2. Jupyter notebook (.ipynb) that contains full markdown sections as listed above 

# SUBMISSION FORMAT FOR THE REPORTS

#  <Title>
**<Brian Robinson, Luiz Salazar, Sam Bryant>** 

## Project Summary

<Complete for *Project Proposal* assignment submission to give idea to the reader what youa re trying to do and how> 
Based on a persons body signals it can be predicted if they are a drinker. This data can be used for alcohol companies to target certain groups.
A person can be classified as a drinker based on information gathered from biometrics.

<Fully rewrite the summary as the last step for the *Project Submission* assignment: github.com repositories on how people shortblurb thre project. It is a standalone section. It is written to give the reader a summary of your work. Be sure to specific, yet brief.>


## Problem Statement 

<Add one sentence for the *Project Proposal* assignment submission that captures the project statement.>
We, as a group, are going to use the dataset to train a model to predict if a person is a drinker or not.

<Expand the section with few sentences for the *Project Progress* assignment submission> 
* Using Body Signals, can a model be created that can predict if a person used to be a drinker or not.
* What is the benchmark you are using.  Why?
We are using this code as our benchmark [code](https://www.kaggle.com/code/raman209/prediction-of-drinkers-using-body-signals) .
This code is very basic and it uses the XGBoost model to acheive accuracy. We are using this as our benchmark because we want to see if we can add features
to improve the accuracy. We will also try out other Machine learning models to reach a higher accuracy. 
* Where does the data come from, what are its characteristics? Include informal success measures (e.g. accuracy on cross-validated data, without specifying ROC or precision/recall etc) that you planned to use. 
The data comes from this [link](https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset/data). It is a dataset that holds body signals and if an indiviudal has smoked or drank in the past. The DRK_YN (Our targte column) is either a Y or N. We decided to make that column binary (0 or 1). The rest of the dataset consists of data and measurements collected for an individual. For example, an individual's height, weight, and age is recored. It also holds the value for their cholesterol and hemoglobin levels. We plan to use a form of cross validation on our data to reduce noise, bias, and variance.
* What do you hope to achieve?
Our goal is to acheive a higher accuracy score than our benchmark. We having already tried normalizing our data with minmax, but it only lowered our accuracy score.
Using only specific columns with high correlation also resulted in an accurcy score lower than 74%.

<Finalize for the *Project Submission* assignment submission> 

## Dataset 

<Add highlights on the dataset, specifically the size in instances and attributes for **Project Proposal**>
[Dataset](https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset/data)
Our dataset is called Smoking and Drinking Dataset with Body Signals. It provides data on an individual's body to predict if they are a drinker (Yes or No) or if they are a smoker(1: Never Smoked, 2: Used to Smoke, 3: Still Smokes). There are 991356 instances and 24 columns in our dataset. 

<Complete the following for the **Project Progress**>
* The dimensions of this dataset is 991346 X 24. There are 24 columns, including the target columns smoker and drinker. 
* 1. Sex: Is an indivudal male or female (1 or 0).
* 2. Age: What is the age of the individual.
* 3. height: The height of an indivuald in meters and centimeters.
* 4. weight: An individual's weight.
* 5. waistline: An individual's waistline measurements. 
* 6. sight_left: Does the individual have left sided sight.
* 7. sight_right: Does the individual have right sided sight.
* 8. hear_left: Does an individual have left sided hearing.
* 9. hear_right: Does an individual have right sided hearing.
* 10. SBP: The Systolic blood pressure.
* 11. DBP: An individual's Diastolic blood pressure[mmHg].
* 12. BLDS: An individual's fasting blood glucose[mg/dL].
* 13. tot_chole: An individual's total cholesterol[mg/dL].
* 14. HDL_chole: An individual's HDL (high-density lipoprotein) cholesterol.
* 15. LDL_chole: An individual's  LDL(low-density lipoproteins) cholesterol.
* 16. triglyceride: An individual's triglyceride[mg/dL].
* 17. hemoglobin: An individual's hemoglobin[g/dL].
* 18. urine_protein: An individual's type of protein in their urine  1(-), 2(+/-), 3(+1), 4(+2), 5(+3), 6(+4).
* 19. serum_creatinine: An individual's  serum(blood) creatinine[mg/dL] levels.
* 20. SGOT_AST: SGOT(Glutamate-oxaloacetate transaminase) AST(Aspartate transaminase)[IU/L].
* 21. SGOT_ALT: ALT(Alanine transaminase)[IU/L].
* 22. gamma_GTP: y-glutamyl transpeptidase[IU/L].
* 23. SMK_stat_type_cd: Smoking state, 1(never), 2(used to smoke but quit), 3(still smoke).
* 24. DRK_YN: Drinker or Not.
* If you are using benchmarks, describe the data in details. If you are collecting data, describe why, how, data format, volume, labeling, etc.>

<Expand and complete for *Project Submission*>

* What Processing Tools have you used.  Why?  Add final images from jupyter notebook. Use questions from 3.4 of the [Datasheets For Datasets](https://arxiv.org/abs/1803.09010) paper for a guide.>  

## Exploratory Data Analysis 

<Complete for **Project Progress**>
* What EDA graphs you are planning to use?
Histograms to view dataset distribution.
Pair plot to visualize relationships between features.
Correlation matrix heatmap to visualize strength of relationship between features.

* Why? - We used pair plot and correlation to figure out what columns we might look at. When experimenting we didn't see a postive effect when trying to train using only certain rows.
We may look to using a dimensionality reduction algorithm to improve results. Histograms allowed us to see the range of values for the age, sex, and drinker. 
Making a histgram for the drinker column allowed us to see that we have a close number of Yes's and No's. Having too much of one prediction would skew results.

<Expand and complete for the **Project Submission**>
* Describe the methods you explored (usually algorithms, or data wrangling approaches). 
  * Include images. 
* Justify methods for feature normalization selection and the modeling approach you are planning to use. 

## Data Preprocessing 

<Complete for *Project Progress*>
* Have you considered Dimensionality Reduction or Scaling? 
  * If yes, include steps here.
  We tried out minmax and it didn't improve the results immediately. We will try to normalize the data differently to get better results.  
* What did you consider but *not* use? Why? 
  We considered using a dimensionality reduction like PCA, but we want to try out scaling the data first and seeing the results. We plan on experimenting
  with PCA or TSNE. 

<Expand and complete for **Project Submission**>


## Machine Learning Approaches

<Complete for **Project Progress**>

* What is your baseline evaluation setup? Why? 
Are baseline evaluation setup is currently checking the accuracy and using a confusion matrix. This will tell us how many false positives and false negeative we have, which will allow us to make changes
to alleviate the error.
* Describe the ML methods that you consider using and what is the reason for their choice?
We will try XGboost because our benchmark is using it. We will try out logistic regression because we are trying to predict something that is binary (Yes or No). We will also try out
an SVM model to compete with the other two.
We plan to compare the two models and try to get a better score than the becnhmark. 
   * What is the family of machine learning algorithms you are using and why?
   We are using models form the decision tree family because we beleive that certain feature decsions could lead to accurate prediction. We are using logistic regression whihc is a part of the linear
   regression tree. logistic regression can be used for binary predictions. We will also try out a model from the SVM family, so we can compare the results to ther other models.

<Expand and complete for **Project Submission**>

* Describe the methods/datasets (you can have unscaled, selected, scaled version, multiple data farmes) that you ended up using for modeling. 

* Justify the selection of machine learning tools you have used
  * How they informed the next steps? 
* Make sure to include at least twp models: (1) baseline model, and (2) improvement model(s).  
   * The baseline model  is typically the simplest model that's applicable to that data problem, something we have learned in the class. 
   * Improvement model(s) are available on Kaggle challenge site, and you can research github.com and papers with code for approaches.  

## Experiments 

< **Project Progress** should include experiments you have completed thus far.>
* We tried XGBoost on the dataset when it was normalized and we tried to pick columns that had higher correlations. We were not able to get a score higher than our benchmark score 74% accuracy.

<**Project Submission** should only contain final version of the experiments. Please use visualizations whenever possible.>
* Describe how did you evaluate your solution 
  * What evaluation metrics did you use? 
* Describe a baseline model. 
  * How much did your model outperform the baseline?  
* Were there other models evaluated on the same dataset(s)? 
  * How did your model do in comparison to theirs? 
  * Show graphs/tables with results 
  * Present error analysis and suggestions for future improvement. 

## Conclusion
<Complete for the **Project Submission**>
* What did not work? 
* What do you think why? 
* What were approaches, tuning model parameters you have tried? 
* What features worked well and what didn't? 
* When describing methods that didn't work, make clear how they failed and any evaluation metrics you used to decide so. 
* How was that a data-driven decision? Be consise, all details can be left in .ipynb

 
## Submission Format
1. Python code with markdown documentation, images saved in .jpg or .png format, and README.md as a project report OR
2. Jupyter notebook (.ipynb) that contains full markdown sections as listed above 

## Now go back and write the summary at the top of the page
# ML-Project
