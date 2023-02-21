import numpy as np
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from datetime import datetime, timedelta
from starlette.responses import JSONResponse
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "https://localhost:3000",
    "http://localhost:3000",
    "https://fntracker.pp.ua",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return 'hello world'


@app.options("/dates")
async def calc_dates3(data: Request):
    return 'gfgfdgfdgdf'


@app.post("/dates")
async def calc_dates(data: Request):
    dates = await data.json()
    dates = dates['dates']
    date_objects = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
    normalized_dates = np.array(
        [(date - min(date_objects)) / (max(date_objects) - min(date_objects)) for date in date_objects])
    inputs = Input(shape=(1,))
    x = Dense(256, activation='relu')(inputs)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(x=normalized_dates[:-1, np.newaxis], y=normalized_dates[1:, np.newaxis], epochs=500, verbose=0)
    predicted_dates = []
    current_date = datetime.now()
    for _ in range(8):
        normalized_current_date = (current_date - min(date_objects)) / (max(date_objects) - min(date_objects))
        predicted_normalized_date = model.predict(np.array([[normalized_current_date]]))[0][0]
        predicted_date = min(date_objects) + timedelta(
            days=predicted_normalized_date * (max(date_objects) - min(date_objects)).days)
        predicted_dates.append(predicted_date.strftime('%Y-%m-%d'))
        current_date = predicted_date
    return JSONResponse(predicted_dates)
