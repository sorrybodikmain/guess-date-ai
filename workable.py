from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from datetime import datetime, timedelta
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Input data
dates = ["2018-02-22", "2018-03-01", "2018-03-07", "2018-04-03", "2018-04-24", "2018-05-17", "2018-08-03", "2018-10-16",
         "2018-11-18", "2018-12-11", "2019-01-18", "2019-05-09", "2019-09-01", "2019-10-07", "2019-12-14", "2020-01-22",
         "2020-03-22", "2020-05-03", "2020-05-27", "2020-06-20", "2020-07-15", "2020-08-01", "2020-08-25", "2020-09-16",
         "2020-10-09", "2020-11-03", "2020-12-05", "2021-01-21", "2021-03-21", "2021-05-11", "2021-06-12", "2021-07-15",
         "2021-08-15", "2021-08-21", "2021-08-22", "2021-09-23", "2021-10-24", "2021-12-10", "2022-01-15", "2022-03-07",
         "2022-04-07", "2022-05-09", "2022-06-15", "2022-08-02", "2022-09-03", "2022-10-05", "2022-11-13"]

print('Convert to datetime objects')
date_objects = [datetime.strptime(date, '%Y-%m-%d') for date in dates]

print('Normalize datetime objects to have a range of [0, 1]')
min_date = min(date_objects)
max_date = max(date_objects)
normalized_dates = [(date - min_date) / (max_date - min_date) for date in date_objects]

print('Define the neural network')
inputs = Input(shape=(1,))
x = Dense(128, activation='relu')(inputs)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(8, activation='linear')(x)

model = Model(inputs=inputs, outputs=outputs)

print('Compile the model')
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

print('Train the model')
model.fit(x=normalized_dates[:-1], y=normalized_dates[1:], epochs=500, verbose=0)
print('End of model training')

print('Predict the next 5 dates')
predicted_dates = []
current_date = date_objects[-1]
for _ in range(10):
    # Normalize the current date
    normalized_current_date = (current_date - min_date) / (max_date - min_date)
    # Predict the next date
    predicted_normalized_date = model.predict([[normalized_current_date]])[0][0]
    # Denormalize the predicted date
    predicted_date = min_date + timedelta(days=predicted_normalized_date * (max_date - min_date).days)
    predicted_dates.append(predicted_date.strftime('%Y-%m-%d'))
    # Update the current date to be the predicted date for the next iteration
    current_date = predicted_date

now = datetime.now()
filtered_dates = filter(
    lambda date: datetime.strptime(date, '%Y-%m-%d') > now, predicted_dates)
print('unfiltered dates--------------')
print(predicted_dates)
print('filtered dates----------------')
for i in filtered_dates:
    print(i)
