# Car-Price-Prediction
Import Libraries: The code begins by importing necessary libraries such as pandas, numpy, matplotlib, seaborn, and TensorFlow.

Read Data: It reads the car price data from the 'train.csv' file using pandas and stores it in the 'data' dataframe.

Data Preprocessing: The data is then converted to TensorFlow tensors and shuffled randomly.

Train-Validation-Test Split: The dataset is split into training, validation, and test sets based on predefined ratios (TRAIN_RETIO, VAL_RETIO, and TEST_RETIO).

Create TensorFlow Datasets: TensorFlow datasets are created from the training and validation data to facilitate batch processing during training.

Normalization: The input data (features) is normalized using TensorFlow's Normalization layer to scale the values between 0 and 1, which helps improve training performance.

Model Architecture: The neural network model is defined using the Sequential API from TensorFlow. The model consists of an input layer, a normalization layer, three hidden Dense layers with ReLU activation functions, and an output Dense layer with one unit for predicting car prices.

Model Compilation: The model is compiled using the Adam optimizer with a learning rate of 0.1 and Mean Absolute Error (MAE) as the loss function. The Root Mean Squared Error (RMSE) is used as an additional metric to monitor model performance during training.

Model Training: The model is trained using the 'fit' method on the training dataset, and the validation dataset is used for validation during training. The training is performed for 100 epochs, and the training and validation loss and RMSE are recorded for each epoch.

Visualization: The training and validation loss are plotted to visualize the model's learning progress over epochs. Similarly, the training and validation RMSE values are plotted to assess the model's performance.

Evaluation: The model is evaluated on the test dataset using the 'evaluate' method, and the results (loss and RMSE) are printed.

Prediction: A single prediction is made using the trained model on the first example in the test dataset, and the actual value from the test dataset is also displayed.

Visualization of Predictions: Finally, a bar chart is created to compare the predicted car prices (y_pred) and the actual car prices (y_true) for the first 100 examples in the test dataset.
