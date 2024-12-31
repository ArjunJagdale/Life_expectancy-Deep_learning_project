# Life Expectancy Prediction Model

This project involves building a simple neural network model to predict life expectancy based on various features using a dataset `life_expectancy.csv`. The code goes through the process of loading, preprocessing, training, and evaluating the model using TensorFlow and scikit-learn. Below is a breakdown of the steps followed in the code.

## Steps:

### 1. Loading the Dataset:
- The dataset is loaded using `pandas.read_csv()` from a CSV file named `life_expectancy.csv`.
- The `head()` and `describe()` methods are used to inspect the first few rows and basic statistics of the dataset.

### 2. Data Preprocessing:
- The `Country` column is dropped from the dataset as it is not needed for prediction.
- The remaining columns are split into features (inputs) and labels (outputs).
- One-hot encoding is applied to the categorical features using `pd.get_dummies()` to convert them into numeric columns suitable for training.

### 3. Splitting the Data:
- The dataset is split into training and testing sets using `train_test_split()` from scikit-learn. 
- The training set comprises 80% of the data, while the test set comprises 20%.

### 4. Standardizing Features:
- A `ColumnTransformer` is used to standardize numerical columns (features with data types `float64` and `int64`) using `StandardScaler()`. The numerical columns are then scaled to have zero mean and unit variance.
- The transformation is applied separately to the training and testing sets.

### 5. Building the Neural Network Model:
- A simple neural network model is created using Keras' `Sequential` API.
- The input layer is created with the number of input features as its size.
- The first hidden layer has 64 neurons with ReLU activation.
- The output layer has a single neuron for predicting life expectancy (a regression task).

### 6. Compiling the Model:
- The model is compiled using the **Adam** optimizer with a learning rate of 0.01.
- The loss function used is **Mean Squared Error (MSE)**, as it is a regression task.
- **Mean Absolute Error (MAE)** is also tracked as an additional metric.

### 7. Training the Model:
- The model is trained for 40 epochs with a batch size of 1. 
- The training progress is shown with verbose output.

### 8. Evaluating the Model:
- After training, the model is evaluated on the test set using the `evaluate()` function, which returns the loss (MSE) and metric (MAE).

### 9. Output:
- The `model.summary()` function displays the architecture of the neural network.
- After evaluation, the Mean Squared Error (MSE) and Mean Absolute Error (MAE) on the test set are printed.

   1/2350 [..............................] - ETA: 0s - loss: 4.2176 - mae: 2.0537
  75/2350 [..............................] - ETA: 1s - loss: 9.3344 - mae: 2.3484
 143/2350 [>.............................] - ETA: 1s - loss: 8.0824 - mae: 2.1808
 214/2350 [=>............................] - ETA: 1s - loss: 10.3103 - mae: 2.3571
 284/2350 [==>...........................] - ETA: 1s - loss: 11.1521 - mae: 2.4512
 358/2350 [===>..........................] - ETA: 1s - loss: 10.6155 - mae: 2.3875


## Technologies Used:
- **Python**: Primary programming language.
- **TensorFlow/Keras**: For building and training the neural network.
- **scikit-learn**: For data preprocessing and splitting the dataset.
- **Pandas**: For data manipulation and cleaning.

## Prerequisites:
- Python 3.x
- Libraries: `pandas`, `scikit-learn`, `tensorflow`
  - Install required libraries via:
    ```bash
    pip install pandas scikit-learn tensorflow
    ```

## Dataset:
The dataset `life_expectancy.csv` contains various features related to countries' life expectancy statistics, with one column being the target variable: **life expectancy**.

---

## Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 64)                1408      
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 65        
=================================================================
Total params: 1,473
Trainable params: 1,473
Non-trainable params: 0

