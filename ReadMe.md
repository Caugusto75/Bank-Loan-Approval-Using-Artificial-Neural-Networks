# Bank Loan Approval Using Artificial Neural Net
## Project objective:
- Understand how Artificial Intelligence and Machine Learning are applied in the banking industry.
- Understand the theory and intuition behind Deep Neural Networks.
- Import key Python libraries, dataset, and perform Exploratory Data Analysis (EDA).
- Perform data visualization using Seaborn.
- Standardize the data and split them into train and test datasets.  
- Build a deep learning model using Keras with Tensorflow 2.0 as a back-end.
- Assess the performance of the model and ensure its generalization using various KPIs.
## Project Structure
### Task 1: Understand the problem statement and business case
- In this hands-on project, we will assume that you work as a data scientist at a bank
- In this project, we aim to build and train a deep neural network model to predict the likelihood of liability customers (depositors) buying personal loans based on customer features such as age, experience, income, location, family, education, existing mortgage, credit cards, etc.
- Lenddo (https://lenddo.com) is a leading startup that uses advanced machine learning to analyze over 12,000+ features from various sources of data to predict an individual’s creditworthiness (e.x: social media count use, geolocation data, etc,)
![](image/Fig1.png)
- Most bank customers are depositors and the bank would like to start targeting those customers and issue personal loans to them
- The bank would like to retain those depositors and convert them to personal loan customers and therefore growing annual revenue
- In the past, the bank ran a marketing campaign that resulted in a 9% success rate.
- The bank management team would like to analyze this campaign results and develop a model to target customers who have high probability of obtaining personal loans.

| Column Name      | Column Description                                                              |
|------------------|---------------------------------------------------------------------------------|
| ID               | Customer ID                                                                     |
| Age              | Customer's age in yearrs                                                        |
| Experience       | Number of years of professional experience                                      |
| Income           | Annual income of customer ($000)                                                |
| Zipcode          | Home Adress Zip Code                                                            |
| Family           | Family size of the customer                                                     |
| CCAvg            | Average spending on credit cards per months ($000)                              |
| Education        | Education level (1: Undergrad;2: Graduated, 3: Advanced/Professional)           |
| Mortgage         | Value of house mortgage ($000)                                                  |
| Personal Loan    | Did this customer accept the personal loan offered in the last campaign or not? |
| Security Account | Does the customer have a securities account with the bank?                      |
| Online           | Does the customer use internet banking facilities?                              |
| Credit Card      | Does the customer uses a credit card issued by UniversalBank                    |

### Task 2: Import Datasets and Libraries
#### Libraries:
- Pandas: Dataframe manipulating library
- Numpy: Numerical values, arrays, matrices handling library 
- Matplotlib & Seaborn: Data visualization libraries
- Tensorflow: Google framework for building and train AI model
#### Dataset
- The dataframe contains 5000 rows

![](image/Bank_df.png)

### Task 3: Exploratory Data Analysis
![](image/Task3-1.png)
- The dataframe contains 14 columns in total
- Customers range in age from 23 to 67, with an average age of 45
- The average of income is $73,774
- The average family size is 2
- Credit Card Debt is an average of $1,938 per person

![](image/Task3-2.png)

- 29.4% of customers in this dataset have credit card debt
- 9.6% of customers in this dataset have a personal loan
### Task 4: Perform Data Visualization
#### Visualize personal Loan column
- Percentage of customers who accepted personal loan ~ 9%

![](image/Task4-1.png)
#### Visualize Education feature 
- Most of customer has undergraduate degree

![](image/Task4-2.png)

#### Visualize Age
- Uniform distribution between 30-60 years

![](image/Task4-3.png)

#### Visualize credit card availability feature
- Recall that ~29% of customers have credit cards

![](image/Task4-4.png)

#### Visualize income data
- Most customers have incomes that range between 45K and 60K per year
- Data is skewed with less customers earning above 100K

![](image/Task4-5.png)

#### Customers with loan vs Customer without loan
- Mean income of customers who have personal loans is generally high ~ 144K and average CC of 3.9K
- Mean income of customers who have do not have personal loans is generally low ~ 66K and average CC of 1.7K

| Average Credit Card Debt  | Average Income           |
|---------------------------|--------------------------|
| ![](image/Task4-6-1.png)  | ![](image/Task4-6-2.png) |
- Customers who took personal loans tend to have higher income & debt

| Average Credit Card Debt | Average Income           |
|---|--------------------------|
| ![](image/Task4-6-3.png)  | ![](image/Task4-6.png) |

#### Correlation plot
- Stong Positive correlation between experience and age
- Strong positive correlation between CC average and income

![](image/Task4-8.png)

#### Credit Card Spending
- Most customers Credit card spending is between 0-4K
- Data is positively skewed

![](image/Task4-9.png)

#### Credit card average and personal loan relationship
- Customers who have large credit card average tend to have personal loans

![](image/Task4-10.png)

### Task 5: Prepare the data to feed the model
#### Features dataframe
- Dataframe X created by  dropping the target variable ‘Personal Loan’

![](image/Task5-1.png)

#### Target dataframe
- Another dataframe y created contains only  target variable 'Personal Loan'

![](image/Task5-2.png)

#### Apply One Hot Encoding Method
- Apply the one hot encoding method to prepare for the algorithm and get a better prediction
- With one-hot, we convert each categorical value into a new categorical column and assign a binary value of 1 or 0 to those columns.

![](image/Task5-3.png)
  
#### Scale and split the data

- Apply the one hot encoding method to prepare for the algorithm and get a better prediction
- - With one-hot, we convert each categorical value into a new categorical column and assign a binary value of 1 or 0 to those columns.
- This is vital to ensure that all the features that are fed to the artificial neural networks have equal weights.
- We must scale the data before training the model using StandardScaler from Sklearn
- - Basically, StandardScaler removes the mean and scales each feature or variable to unit variance
- Split the data into testing and training sets 
- - 90% for trainning
- - 10% for testing

![](image/Task5-4.png)

### Task 6: Understand the theory and intuition behind Artificial Neural Networks
#### Neuro Mathematical Model
- Using an input channel called dendrites, the neuron collects signals, processes them in its nucleus, and then generates an output in the form of long thin branches called axons.

![](image/Task6-1.png)

#### Single neuron model math
- Bias allows to shift the activation function curve up or down.
- Number of adjustable parameters = 4 (3 weights and 1 bias).
- Activation function F.

![](image/Task6-2.jpeg "Single Neuron Model Math")

#### ANN Training Process
- **Inputs X**, arrive through the preconnected path
- Input is modeled using real weights W. The **weights are usually randomly selected**.
- **Calculate** the output for every neuron from the input layer, to the hidden layers, to the **output layer**.
- **Calculate the error in the outputs**
- **Compute the gradient to change the weight:** Travel back from the output layer to the hidden layer to adjust the weights such that the error is decreased.
- Keep repeating the process until the desired output is achieved (**the loss function is stable: Optimization done!**)

| ANN Training Process                               | Error Performance Measure vs Epoch of Times |
|----------------------------------------------------|---------------------------------------------|
| ![](image/Task6-5.webp)| ![](image/Task6-4.png "Loss V.s Epoch of Times")                      |
#### Gradient Descent
- Gradient descent is an optimization algorithm used to obtain the optimized network weight and bias values
- It works by iteratively trying to minimize the const function
- It works by calculating the gradient of the cost function and moving in the negative direction until the local/global minimum is achieved
- If the positive of the gradient is taken, local/global maximum is achieved

Learning Rate
- The size of the steps taken are called the learning rate
- The learning rate increases, the area covered in the search space will increase so we might reach  global minimum faster
- However, we can overshoot the target
- For small learning rates, training will take much longer to reach  optimized weight values




### Task 7: Build a simple Multi Layer Neural Network
### Task 8: Compile and train a Deep Learning Model
### Task 9: Assess the performance of the trained model

