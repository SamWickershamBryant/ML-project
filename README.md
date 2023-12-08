
#  Predicting Whether A Person Is A Drinker Based On Body Signals
**Brian Robinson, Sam Bryant** 

## Project Summary

### Introduction

Can we predict if someone is a drinker based on their body signals? In this project we will be using a kaggle data from [here](https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset) that contains almost a million instances collected from the National Health Insurance Service in Korea. It contains whether the person is a drinker or not, along with over 20 different body signals including age, sex, weight, height, and others. We will use this dataset to create an accurate model for predicting someone's drinking status based off the given body signals.

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
This code is very basic and it uses the XGBoost model to achieve accuracy. We are using this as our benchmark because we want to see if we can add features
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


## Data Preprocessing

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



### Planned EDA Graphs:
- **Histogram for Age Distribution**: This graph will be used to understand the distribution and range of ages in the dataset, which is critical for assessing demographic factors.
- **Correlation Heatmap**: A heatmap will be created to visually represent the correlation between different variables. This is essential for identifying potential predictors for the model.
- **Pair Plot**: To explore pairwise relationships and distributions, a pair plot will be used. It's an effective way to spot trends, outliers, and patterns across multiple dimensions.

**Rationale Behind Graph Selection:**
- The histogram is a fundamental tool for examining the distribution of a single variable and can reveal skewness or outliers.
- Correlation heatmaps are crucial in understanding the relationships between variables, highlighting potential dependencies or multicollinearity.
- Pair plots provide a comprehensive overview of how each variable relates to others in a dataset, offering insights into complex interactions.


### Methods Explored and Justification:
1. **Drinker Distribution Analysis**:
   - We used a drinker distribution histrogram to analyze our target variable. This information was important for understanding if we had to use any data engineering techniques to balance the data.
   - Image: ![Drinker Distribution Histogram](https://i.imgur.com/TMV12QL.png)
   
2. **Age Distribution Analysis**:
   - The histogram of age was used to analyze the demographic distribution within the dataset. Understanding age distribution is vital for tailoring further analysis and models to the specific age groups present in the data.
   - Image: ![Age Distribution Histogram](https://i.imgur.com/f8cHcL7.png)

3. **Correlation Analysis**:
   - A correlation heatmap was created to identify and visualize the strength and direction of relationships between the variables. This step is critical for feature selection and helps in avoiding features that are highly correlated with each other.
   - Image: ![Correlation Heatmap](https://i.imgur.com/C6odaax.png)

4. **Pair Plot**:
   - The pair plot was used to inspect the pairwise relationships and distributions among the variables. It assists in identifying any specific trends or anomalies in the dataset and is useful for preliminary feature selection.
   - Image: ![Pair Plot](https://i.imgur.com/PKYgrXo.png)

### Feature Normalization Selection and Modeling Approach:
- The choice of feature normalization and the modeling approach will be based on insights derived from these EDA methods. For instance, the distribution of variables as indicated by the histogram can guide the choice of normalization technique. Similarly, understanding correlations and relationships will influence the modeling approach, especially in terms of feature selection and engineering.

## Dimensionality Reduction and Scaling

- **Scaling**: Scaling was considered essential for normalizing the feature scales in the dataset. This step is critical to ensure that models that are sensitive to the scale of data, such as PCA, perform optimally.
- **Dimensionality Reduction**: Dimensionality reduction, specifically using PCA (Principal Component Analysis), was planned to reduce the number of features while retaining the maximum variance in the data.

**Considered but Not Used:**
- **TSNE**: t-Distributed Stochastic Neighbor Embedding (t-SNE) was considered but not used. We determined that t-SNE, while excellent for visualization, is not as effective for our modeling purposes, as it is primarily designed for high-dimensional data visualization and not for improving model accuracy.
- **Min-Max Scaling**: This technique was explored but ultimately not used. We found that Min-Max scaling adversely affected our model's accuracy in preliminary tests.


### Implemented Methods:
1. **Feature Scaling with StandardScaler**:
   - Applied StandardScaler to normalize the features in our dataset. This step is crucial for models that assume normally distributed data.
   - ![Standard Scaler Output](https://i.imgur.com/ynFC8yV.png)

2. **PCA for Dimensionality Reduction**:
   - PCA was used after scaling to reduce the dataset's dimensionality. We aimed to retain features that explain up to 95% of the variance in the dataset.
   - The PCA cumulative variance plot helped determine the number of components needed.
   - ![PCA Cumulative Variance Plot](https://i.imgur.com/2C0dJri.png)
   - The optimal number of PCA components was calculated to effectively capture the significant variance in the data, determined to be `17` components.
   - ![PCA Component Analysis](https://i.imgur.com/MLnbCCI.png)

### Justification for Method Selection:
- StandardScaler was chosen as it standardizes features by removing the mean and scaling to unit variance, which is particularly beneficial for PCA.
- PCA was selected for its effectiveness in reducing dimensionality while preserving as much information as possible. The choice of 95% variance retention was a balance between complexity reduction and information retention.



## Machine Learning Approaches


### Baseline Evaluation Setup:
- **Method**: Our baseline evaluation setup includes checking the model's accuracy and using a confusion matrix. This approach helps us understand the model's performance in terms of false positives and false negatives, which is crucial for improving prediction accuracy in a binary classification problem. Model accuracy also gives a direct number that we can use to easily determine if a model is performing well.

![Confusion Matrix](https://i.imgur.com/x2kI3sO.png)


### Considered Machine Learning Methods:
- **XGBoost**: Selected to use as it's in our benchmark, known for its effectiveness in various classification tasks.
- **Logistic Regression**: Chosen for its suitability in binary classification problems, which aligns with our objective of predicting a binary outcome.
- **Random Forest**: Also under consideration as it falls under the decision tree family, believed to be effective in capturing feature-based decisions.

**Family of Algorithms:**
- **Decision Trees**: Decision tree models are being used because of their ability to make accurate predictions based on feature decisions.
- **Logistic Regression**: Part of the linear regression family, it's particularly suited for binary predictions.


### Methods and Datasets Used for Modeling:
- We employed datasets in scaled, unscaled, and reduced forms to determine which one improves our model and by how much.
- Multiple data frames were utilized to explore different aspects and combinations of the data.

**Justification of Machine Learning Tool Selection:**
- The chosen tools informed our next steps by highlighting areas of strength and weakness in our prediction capabilities, guiding adjustments and refinements in our models.
- We incorporated feedback from each model's performance to iteratively improve our approach.

### Model Comparison:
1. **Baseline Model:**
   - **Decision Tree**: Used as our baseline model, it represents the simplest applicable model for our data problem, providing a benchmark for comparison.
2. **Improvement Models:**
   - **XGBoost, Logistic Regression, Random Forest**: These models were selected for their potential to outperform the baseline model. They were tested both with and without the application of the Standard Scaler.
   - The improvement models were inspired by approaches found on Kaggle, as well as what we learned in class.


## Experiments


### Experiments Conducted:
- **XGBoost with Normalized Data**: Applied XGBoost to the dataset after normalization and selected columns with higher correlations. 
- **PCA Dimensionality Reduction**: Attempted to use PCA in order to improve model performance and reduce overfitting.
- **Hyperparameter Tuning**: Used Hyperparameter Tuning to improve our highest accuracy model by choosing parameter that best fit the dataset.
- **Outcome**: The accuracy achieved was below our benchmark score, peaking at 73.40%. 


### Evaluation of Solution:
- **Primary Metric**: The primary evaluation metric used was the accuracy score.
- **Additional Metric**: An AUC (Area Under the Curve) graph was utilized to evaluate model performance.
  - ![AUC Graph](https://i.imgur.com/GIHZksV.png)

**Baseline Model Comparison:**
- **Model Performance**: Our model showed an improvement of approximately 2.2% over the baseline model.
- **Baseline Model Details**: The baseline was a basic decision tree algorithm with default parameters.

**Comparison with Other Models:**
- **Benchmark Model**: The benchmark model, also using XGBoost, was evaluated on the same dataset.
- **Performance Comparison**: Our model underperformed compared to the benchmark by about 0.6%.
- **Result Visualization**:
  - Benchmark model: ![Comparison Graph](https://i.imgur.com/3NnwnIM.png)
  - Our Model: ![Our Graph](https://i.imgur.com/x2kI3sO.png)

**Error Analysis and Future Improvement Suggestions:**
- **Error Analysis**: Our model failed at all of our experiments when trying to outperform the benchmark. When we tried PCA, it just hurt our model. When we tried hyperparameter tuning, it had no effect. The only thing that helped us was using standard scaling but this is to be expected when working with unnormalized data in classification tasks.
- **Improvement Suggestions**: We could improve our performance on learning this data by using different classification models we haven't tried, and by using more extensive hyperparameter tuning. The tuning we used in this project was very basic in order to save on computation time which is probably the reason it didn't improve anything. We could also explore alternative feature engineering techniques or additional data integration to help our model, but this is not guranteed as the dataset given is already very diverse and extensive.

## Conclusion

PCA, correlation matrix, minmaxscaler did not work to improve our model. We saw no improvements from excluding data in any way. We beleive that these tools did not work because the data is already well balanced and diverse, so any loss of data or change to it is detrimental to the accuracy of the models. This is because every feature is used to determine the target coulumn and models like xgBoost can train effectively using boosting. We tried using grid search on our xgBoost model to tune the parameters. The best parameters found were only able to acheive a score of 72%, which was worse than xgBoost with default parameters. The new models, logistic regression, random forest, and xgBoost, improved our overall score. Training other models improved the accuracy score that we could acheive. Logisitic regression, random forest, and xgBoost were all able to get a better score than our baseline decision tree. In conclusion, we were not able to get an accuracy score over 74%. Despite not reaching our goal, we still were able to improve from our baseline.

 
## Submission Format
1. Python code with markdown documentation, images saved in .jpg or .png format, and README.md as a project report OR
2. Jupyter notebook (.ipynb) that contains full markdown sections as listed above 

