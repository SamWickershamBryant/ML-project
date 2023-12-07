# Project Assignments 

* [Project Proposal](https://canvas.txstate.edu/courses/2179554/assignments/31569710) 
* [Project Progress](https://canvas.txstate.edu/courses/2179554/assignments/31569830)
* [Project Submission](https://canvas.txstate.edu/courses/2179554/assignments/31569848)

 **Submission Format** 
 
1. Python code with markdown documentation, images saved in .jpg or .png format, and README.md as a project report OR
2. Jupyter notebook (.ipynb) that contains full markdown sections as listed above 

# SUBMISSION FORMAT FOR THE REPORTS

#  Determine if someone is a drinker based on bady signals
**Brian Robinson, Sam Bryant** 

## Project Summary

### Introduction

Can we predict if someone is a drinker based on their body signals? In this project we will be using a kaggle data from [https://www.kaggle.com/code/raman209/prediction-of-drinkers-using-body-signals](https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset) that contains almost a million instances collected from the National Health Insurance Service in Korea. It contains whether the person is a drinker or not, along with over 20 different body signals including age, sex, weight, height, and others. We will use this dataset to create an accurate model for predicting someone's drinking status based off the given body signals.

### Machine Learning Concepts Used

- **Correlation Matrix**: Simple chart for visualizing correlation between features.
- **Cross Validation**: Statistical method used to estimate the skill of machine learning models by partitioning data into subsets, training the model on some subsets and testing it on the remaining ones, thereby helping to mitigate overfitting and providing a more accurate measure of a model's predictive performance.
- **Decision Tree**: Model for making preditions by creating splits on features that can determine the result while being easy to comprehend
- **XGBoost**: Efficient and scalable machine learning model for gradient boosting, known for its performance in classification, regression, and ranking tasks.
- **Logistic Regression**: Statistical model used for binary classification that estimates the probability of a binary response based on one or more predictor variables.
- **Random Forest**: Versatile and robust ensemble learning method that constructs multiple decision trees during training and outputs the class that is the mode of the classes of the individual trees.
- **Principal Component Analysis (PCA)**: Dimensionality reduction technique that transforms data into a new coordinate system, reducing the number of variables while preserving as much variance as possible.
- **Hyperparameter Tuning**: Systematically searching for the optimal set of hyperparameters that governs the learning process of a machine learning algorithm to enhance its performance on a given dataset.


## Problem Statement 

Can we, as a group outperform a benchmark model performance by using these machine learning techniques?


We are using this [code](https://www.kaggle.com/code/raman209/prediction-of-drinkers-using-body-signals) as our benchmark  .
This code is very basic and it uses the XGBoost model to acheive accuracy. We are using this as our benchmark because we want to see if we can add features
to improve the accuracy. We will also try out other Machine learning models to reach a higher accuracy. 
The data comes from this [link](https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset/data). The data was gathered in Korea from a National Health Insurance service. The dataset consists of body signals and if an indiviudal was a smoker or drinker. The DRK_YN (Our targte column) is either a Y or N. We decided to make that column binary (0 or 1). The rest of the dataset contains measurements collected for an individual. For example, an individual's height, weight, and age is recored. It also holds the value for their cholesterol and hemoglobin levels. We plan to use a form of cross validation on our data to reduce noise, bias, and variance. We can use this as an informal success measure.

### What we hope to achieve
Our goal is to acheive a higher accuracy score than our benchmark. The accuracy score of 74% turned out to be difficult to beat but not too hard to get close to. If we are unable to beat this score, we can still learn and make conclusions base on the variable importance derived from the models we use.


## Dataset 

[Dataset](https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset/data)
Our dataset is called Smoking and Drinking Dataset with Body Signals. It provides data on an individual's body to predict if they are a drinker (Yes or No) or if they are a smoker(1: Never Smoked, 2: Used to Smoke, 3: Still Smokes). There are 991356 instances and 24 columns in our dataset. 

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

We are using a benchmark to compare our score with. A basic XGBoost model was able to get an accuracy score of 74%. We plan to build off of this benchmark and add features to get a higher score.
We will not be collecting data.

### Data Processing

Here's an example of how you could document the data processing steps you've described, using Markdown. This format includes placeholders for images of the code/output, which you can replace with actual images or links to images as needed.

---

## Data Preprocessing Documentation

### Preprocessing/Cleaning/Labeling of Data
The dataset underwent several preprocessing steps to prepare it for analysis. These steps include:

1. **Reading the Data**: The dataset, named `smoking_driking_dataset_Ver01.csv`, was read into a pandas DataFrame. 
   - Code Snippet: 
     ```python
     df = pd.read_csv('smoking_driking_dataset_Ver01.csv')
     df.head(5)
     ```
   - Output: ![Data Head](https://i.imgur.com/lmeMUHa.png)

2. **Checking Dimensions**: The dimensions of the dataset were inspected to understand its size.
   - Code Snippet:
     ```python
     df.shape
     ```
   - Output: ![Data Shape](https://i.imgur.com/wFAoM5w.png)

3. **Handling Missing Values**: Checked for and processed any missing values in the dataset.
   - Code Snippet:
     ```python
     df.isna().sum()
     ```
   - Output: ![Missing Values](https://i.imgur.com/SRuPQpe.png)

4. **Column Inspection**: Examined the dataset's columns to understand the data types and structure.
   - Code Snippet:
     ```python
     df.info()
     ```
   - Output: ![Column Info](https://i.imgur.com/hyFIFpu.png)

5. **Dropping Unnecessary Columns**: Removed the 'SMK_stat_type_cd' column as it was not relevant to our analysis.
   - Code Snippet:
     ```python
     df.drop(columns=['SMK_stat_type_cd'], inplace=True)
     ```

6. **Binary Encoding**: Converted categorical columns like 'sex' and 'DRK_YN' to binary values for analysis.
   - Code Snippet:
     ```python
     df['sex'] = (df['sex'] == 'Male').astype(int)
     df['DRK_YN'] = (df['DRK_YN'] == 'Y').astype(int)
     df.head(5)
     ```
   - Output: ![Binary Encoding](https://i.imgur.com/8kMjR7I.png)

7. **Separating Features and Target Variable**: Isolated the target variable 'DRK_YN' from the feature set.
   - Code Snippet:
     ```python
     y = df['DRK_YN']
     x = df.drop('DRK_YN', axis=1)
     x.head(10)
     ```
   - Output: ![Features and Target](https://i.imgur.com/g72J55m.png)

### Raw Data Availability
- The “raw” data, prior to preprocessing, has been preserved to support future analysis and use cases.
- Access Link: [Raw Dataset Link](https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset)

### Availability of Preprocessing Software
- The preprocessing was conducted using standard Python libraries (`numpy`, `pandas`, `seaborn`, `sklearn`, `matplotlib`).
- These libraries are widely available and can be installed via pip.

### Additional Comments
- The preprocessing steps were tailored to ensure the dataset is suitable for binary classification analysis.
- The focus was on simplifying the dataset and transforming categorical variables into a format conducive for machine learning algorithms.


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
Our baseline evaluation setup is currently checking the accuracy and using a confusion matrix. This will tell us how many false positives and false negeative we have, which will allow us to make changes
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
