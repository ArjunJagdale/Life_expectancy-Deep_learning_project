import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load the dataset
dataset = pd.read_csv('life_expectancy.csv')

# Display the first few rows of the dataset
print(dataset.head())

# Display summary statistics of the dataset
print(dataset.describe())

# Drop the 'Country' column as it is not needed for the model
dataset = dataset.drop(['Country'], axis=1)

# Separate the features (input variables) and labels (target variable)
labels = dataset.iloc[:, -1]  # The last column is the label
features = dataset.iloc[:, 0:-1]  # All columns except the last one are features

# Apply one-hot encoding to categorical features
features = pd.get_dummies(features)

# Split the data into training and testing sets
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.20, random_state=23
)

# Select only numerical columns for scaling
numerical_features = features.select_dtypes(include=['float64', 'int64'])
numerical_columns = numerical_features.columns

# Standardize numerical features
ct = ColumnTransformer([('only numeric', StandardScaler(), numerical_columns)], remainder='passthrough')

# Apply scaling to the training and testing sets
features_train_scaled = ct.fit_transform(features_train)
features_test_scaled = ct.transform(features_test)

# Initialize the neural network model
model = Sequential()

# Add an input layer with the shape of the input features
input = InputLayer(input_shape=(features.shape[1],))
model.add(input)

# Add a hidden layer with 64 neurons and ReLU activation function
model.add(Dense(64, activation='relu'))

# Add an output layer with 1 neuron (for regression)
model.add(Dense(1))

# Print the summary of the model architecture
print(model.summary())

# Initialize the Adam optimizer with a learning rate of 0.01
opt = Adam(learning_rate=0.01)

# Compile the model with Mean Squared Error (MSE) loss and Mean Absolute Error (MAE) metrics
model.compile(loss='mse', metrics=['mae'], optimizer=opt)

# Train the model on the training data for 40 epochs with a batch size of 1
model.fit(features_train_scaled, labels_train, epochs=40, batch_size=1, verbose=1)

# Evaluate the model on the testing data
res_mse, res_mae = model.evaluate(features_test_scaled, labels_test, verbose=0)

# Print the evaluation results
print(res_mse, res_mae)

# Print the summary of the model architecture again
print(model.summary())
