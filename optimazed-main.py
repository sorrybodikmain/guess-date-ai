import numpy as np
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from datetime import datetime, timedelta
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Input data
dates = ["2020-03-22", "2020-03-23", "2020-04-26", "2020-05-25", "2020-06-26", "2020-07-28", "2020-09-01", "2020-10-05",
         "2020-11-07", "2020-12-09", "2021-02-25", "2021-03-19", "2021-04-21", "2021-05-26", "2021-05-27", "2021-05-28",
         "2021-05-29", "2021-05-30", "2021-05-31", "2021-07-13", "2021-08-19", "2021-09-22", "2021-11-07", "2021-12-08",
         "2022-01-25", "2022-03-01", "2022-04-06", "2022-05-09", "2022-06-16", "2022-07-20", "2022-08-29", "2022-11-11",
         "2022-12-11", "2023-01-25", "2023-01-26", "2023-02-14", "2023-02-15", "2023-02-16", "2023-02-17", "2023-02-18",
         "2023-02-19", "2023-02-20"]

# Convert to datetime objects and normalize to have a range of [0, 1]
date_objects = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
normalized_dates = np.array(
    [(date - min(date_objects)) / (max(date_objects) - min(date_objects)) for date in date_objects])

# Define the neural network
inputs = Input(shape=(1,))
x = Dense(256, activation='relu')(inputs)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(1, activation='linear')(x)
model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train the model
model.fit(x=normalized_dates[:-1, np.newaxis], y=normalized_dates[1:, np.newaxis], epochs=500, verbose=0)

# Predict the next 5 dates
predicted_dates = []
current_date = datetime.now()
for _ in range(5):
    # Normalize the current date
    normalized_current_date = (current_date - min(date_objects)) / (max(date_objects) - min(date_objects))

    # Predict the next date
    predicted_normalized_date = model.predict(np.array([[normalized_current_date]]))[0][0]

    # Denormalize the predicted date
    predicted_date = min(date_objects) + timedelta(
        days=predicted_normalized_date * (max(date_objects) - min(date_objects)).days)
    predicted_dates.append(predicted_date.strftime('%Y-%m-%d'))

    # Update the current date to be the predicted date for the next iteration
    current_date = predicted_date

print(predicted_dates)
