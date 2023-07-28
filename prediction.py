import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.callbacks import ReduceLROnPlateau

#csv 파일 읽어 dataset 생성
def ReadDibCsv(count, filepath): 
    dataset = np.zeros((count, 68, 2)) # 3rd dim
    for i in range(count):
        df = pd.read_csv(filepath +'%s.csv'%i)
        dataset[i] = df.to_numpy()
    return dataset


# 기본 좌표를 이용해 눈, 코, 입, 턱, 머리 데이터 나누기
def Standard(dataset):
    cnt = len(dataset)

    pos = np.zeros((cnt, 4))
    eyes = np.zeros((cnt, 24))
    nose = np.zeros((cnt, 18))
    mouth = np.zeros((cnt, 40))
    jaws = np.zeros((cnt, 34))
    head = np.zeros((cnt, 34))

    for i in range(cnt):
        x = dataset[i, :, 0]
        y = dataset[i, :, 1]
        stack = np.column_stack((x, y))

        eye_center = y[42:48].mean()
        mouth_center = y[48:].mean()

        pos[i][0] = eye_center
        pos[i][1] = y[30]
        pos[i][2] = mouth_center
        pos[i][3] = y[8]

        eyes[i] = stack[36:48].flatten()
        nose[i] = stack[27:36].flatten()
        mouth[i] = stack[48:].flatten()
        jaws[i] = stack[:17].flatten()
        head[i] = stack[:17].flatten()

    return pos, eyes, nose, mouth, jaws, head


# 매핑 방식을 이용해 정규화 후 데이터 나누기
def Norm_map(dataset):
    cnt = len(dataset)

    pos = np.zeros((cnt, 4))
    eyes = np.zeros((cnt, 24))
    nose = np.zeros((cnt, 18))
    mouth = np.zeros((cnt, 40))
    jaws = np.zeros((cnt, 34))
    head = np.zeros((cnt, 34))

    for i in range(cnt):
        x = dataset[i, :, 0]
        y = dataset[i, :, 1]

        eye_center = y[42:48].mean()
        peh = y[8] - eye_center
        rate = (3.0 * peh) / 5.4      ## 임의 설정

        x *= rate
        y *= rate

        stack = np.column_stack((x, y))

        eye_center = y[42:48].mean()
        mouth_center = y[48:].mean()

        pos[i][0] = eye_center
        pos[i][1] = y[30]
        pos[i][2] = mouth_center
        pos[i][3] = y[8]

        eyes[i] = stack[36:48].flatten()
        nose[i] = stack[27:36].flatten()
        mouth[i] = stack[48:].flatten()
        jaws[i] = stack[:17].flatten()
        head[i] = stack[:17].flatten()

    return pos, eyes, nose, mouth, jaws, head


# y축을 기준으로 정규화 후 데이터 나누기
def Norm_y(dataset):
    cnt = len(dataset)

    pos = np.zeros((cnt, 4))
    eyes = np.zeros((cnt, 24))
    nose = np.zeros((cnt, 18))
    mouth = np.zeros((cnt, 40))
    jaws = np.zeros((cnt, 34))
    head = np.zeros((cnt, 34))

    for i in range(cnt):
        x = dataset[i, :, 0]
        y = dataset[i, :, 1]

        x -= ((min(x) + max(x)) / 2)
        y -= ((min(y) + max(y)) / 2)

        height = (max(y) - min(y))
        norm_x = (x - min(x)) / height
        norm_y = (y - min(y)) / height
        stack = np.column_stack((norm_x, norm_y))

        eye_center = norm_y[42:48].mean()
        mouth_center = norm_y[60:68].mean()

        pos[i][0] = eye_center
        pos[i][1] = y[30]
        pos[i][2] = mouth_center
        pos[i][3] = y[8]
    
        eyes[i] = stack[36:48].flatten()
        nose[i] = stack[27:36].flatten()
        mouth[i] = stack[48:].flatten()
        jaws[i] = stack[:17].flatten()
        head[i] = stack[:17].flatten()

    return pos, eyes, nose, mouth, jaws, head


# LinearRegression predict
def LinearTrain(train_x, train_y, test_x):
    model = LinearRegression()
    model.fit(train_x, train_y)
    joblib.dump(model, './model.pkl')
    pred = model.predict(test_x)

    return pred


# MLP predict
def MLPTrain(train_x, train_y, test_x, test_y):
    np.random.seed(1)
    tf.random.set_seed(1)

    learning_rate = 0.0005
    N_EPOCHS = 500
    N_BATCH = 32

    reduceLR = ReduceLROnPlateau(monitor='val_loss',
                            factor=0.5,
                            patience=10,
                            min_delta=0.0001,
                            cooldown=0,
                            min_lr=0,
                            mode='auto')
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))\
                                .shuffle(500)\
                                .batch(N_BATCH, drop_remainder=True)\
                                .repeat()
    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(N_BATCH)

    ## model 생성
    model = Sequential()
    model.add(layers.Dense(units=32, activation='relu', input_shape=len(train_x[0])))
    model.add(layers.Dense(units=16, activation='relu'))
    model.add(layers.Dense(units=len(train_y[0])))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')

    steps_per_epoch = train_x.shape[0]
    validation_steps = int(np.ceil(test_x.shape[0]/N_BATCH))

    history = model.fit(train_dataset,
                        epochs=N_EPOCHS,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=test_dataset,
                        validation_steps=validation_steps,
                        callbacks=[reduceLR])
    
    pred = model.predict(test_x)

    return pred


# 예측값 실제값 비교(mse, mae)
def evaluate(test_y, pred):
    mse = mean_squared_error(test_y, pred)**0.5
    mae = mean_absolute_error(test_y, pred)

    return mse, mae