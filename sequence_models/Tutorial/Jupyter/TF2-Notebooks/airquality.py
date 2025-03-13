import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

if __name__ == '__main__':
    # Dataset: https://archive.ics.uci.edu/ml/datasets/air+quality
    path = os.path.join(os.curdir, 'lego_dataset', 'AirQualityUCI', 'AirQualityUCI.xlsx')
    dataset = pd.read_excel(path)
    dataset['date_time'] = pd.to_datetime(dataset['Date'].astype(str) + ' ' + dataset['Time'].astype(str))
    # Time Format: 2004-03-10 18:00:00
    dataset.set_index('date_time', inplace=True)
    dataset.sort_index(ascending=True, inplace=True)
    dataset.drop(['Date', 'Time'], inplace=True, axis=1)
    # Some of the temperature entries are negative, setting them to Zero
    dataset.loc[dataset['T'] < 0, 'T'] = 0

    # Check for the NULL values
    null = pd.DataFrame(dataset.isnull().sum()).rename(columns={0: "Total"})
    null['percentage'] = null['Total'] / len(dataset)
    null.sort_values('percentage', ascending=False).head()
    print(null.head())

    # Since the conditions are periodic
    timestamp_s = dataset.index.map(pd.Timestamp.timestamp)
    day = 24 * 60 * 60
    dataset['Day_Sin'] = np.sin(timestamp_s * 2 * np.pi / day)
    dataset['Day_Cos'] = np.cos(timestamp_s * 2 * np.pi / day)
    print(dataset.head())

    train = dataset[0: int(0.7 * len(dataset))]
    val = dataset[int(0.7 * len(dataset)):int(0.9 * len(dataset))]
    test = dataset[int(0.9 * len(dataset)):]

    train_label = train['T']
    train_data = train.drop(['T'], axis=1)

    val_label = val['T']
    val_data = val.drop(['T'], axis=1)

    test_data = test.drop(['T'], axis=1)
    test_label = test['T']

    mean = train.mean()
    std = train.std() + 1e-12

    train = (train - mean) / std
    val = (val - mean) / std
    test = (test - mean) / std

    print(train.shape)
    print(train.head())

    # Creating dataloader == Non Sequntial loader
    def non_sequential_train_loader(data, labels, batchsize=32, buffersize=100):
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))
        dataset = dataset.cache().shuffle(buffersize).batch(batchsize)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset


    def non_sequential_val_loader(data, labels, batchsize=32):
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))
        dataset = dataset.cache().batch(batchsize)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset


    def non_sequential_test_loader(data, batchsize=32):
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.cache().batch(batchsize)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset


    train_load = non_sequential_train_loader(train.to_numpy(), train_label.to_numpy())
    val_load = non_sequential_val_loader(val.to_numpy(), val_label.to_numpy())
    test_load = non_sequential_test_loader(test.to_numpy())

    for instance, labels in train_load.take(1):
        print(instance.shape)
        print(labels.shape)

    features = 15
    classes = 1
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=[features]),
        tf.keras.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
        tf.keras.layers.Dense(classes)
    ])

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['mse'])
    print(model.summary())

    earlystop = tf.keras.callbacks.EarlyStopping(patience=10)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('best_air/', save_best_only=True)

    # model.fit(train_load, validation_data=val_load, epochs=200, callbacks=[earlystop, checkpoint])

    model = tf.keras.models.load_model('best_air/')

    prediction = model.predict(test_load)
    print(prediction.shape)
    print(test_label.to_numpy().shape)
    print(f'MSE for Nonsequential model: {tf.keras.metrics.mean_squared_error(tf.squeeze(prediction, axis=-1), test_label.to_numpy())}')

    # LEts try sequential model
    def create_sequential_train_loader(series, window_size=24, batchsize=32, buffersize=100):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        dataset = tf.data.Dataset.from_tensor_slices(series)
        dataset = dataset.window(window_size + 1, drop_remainder=True, shift=1)
        dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
        dataset = dataset.map(lambda window: (window[:-1, :-1], window[-1, -1]), num_parallel_calls=AUTOTUNE)
        dataset = dataset.cache().shuffle(buffersize).batch(batchsize)
        return dataset.prefetch(tf.data.experimental.AUTOTUNE)


    def create_sequential_val_loader(series, window_size=24, batchsize=32):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        dataset = tf.data.Dataset.from_tensor_slices(series)
        dataset = dataset.window(window_size + 1, drop_remainder=True, shift=1)
        dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
        dataset = dataset.map(lambda window: (window[:-1, :-1], window[-1, -1]), num_parallel_calls=AUTOTUNE)
        dataset = dataset.cache().batch(batchsize)
        return dataset.prefetch(tf.data.experimental.AUTOTUNE)


    train = tf.concat([train.to_numpy(), train_label.to_numpy().reshape(-1, 1)], axis=1)
    val = tf.concat([val.to_numpy(), val_label.to_numpy().reshape(-1, 1)], axis=1)
    train = create_sequential_train_loader(train)
    val = create_sequential_val_loader(val)
    for X, Y in train.take(1):
        print(X.shape)
        print(Y.shape)

    features = 15
    classes = 1
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=[None, features]),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(classes),
    ])
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['mse'])
    print(model.summary())

    earlystop = tf.keras.callbacks.EarlyStopping(patience=10)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('best_air_sequential/', save_best_only=True)

    model.fit(train, validation_data=val, epochs=200, callbacks=[earlystop, checkpoint])

    model = tf.keras.models.load_model('best_air_sequential/')


    def forecasting(model, series, window_size=24, batchsize=32):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        dataset = tf.data.Dataset.from_tensor_slices(series)
        dataset = dataset.window(window_size, drop_remainder=True, shift=1)
        dataset = dataset.flat_map(lambda window: window.batch(window_size))
        dataset = dataset.cache().batch(batchsize).prefetch(AUTOTUNE)
        return model.predict(dataset)


    window_size = 24
    prediction = forecasting(model, series=test.to_numpy(), window_size=window_size)
    target = test_label.to_numpy()[window_size - 1:]  # Because all the previous values will be dropped!
    print(prediction.shape)
    print(target.shape)
    print(f'MSE for Sequential model: {tf.keras.metrics.mean_squared_error(tf.squeeze(prediction, axis=-1), target)}')