import ast

import tensorflow as tf
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':

    # Dataset: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
    dataset = os.path.join(os.curdir, 'lego_dataset', 'breast-cancer-wisconsin.data')
    with open(dataset, 'r') as file:
        result = file.readlines()
    # print(result)

    processed_result = []
    for instance in result:
        string_instance = instance.replace('\n', '')
        temp = ast.literal_eval(string_instance.replace('?', '0'))
        processed_result.append(list(temp))

    data = pd.DataFrame(np.asarray(processed_result), columns=['id', 'A', 'B', 'C', 'D', 'E',
                                                               'F', 'G', 'H', 'I', 'target'])
    data['target'] = data.target.replace({2: 1, 4: 0})  # Giving the proper targets

    # data['target'].value_counts().plot(kind='bar', figsize=(10, 10))
    # plt.show()

    print(data.describe())
    print(data.info())

    null = pd.DataFrame(data.isnull().sum()).rename(columns={0: 'Total'})
    null['percentage'] = null['Total'] / len(data)
    null.sort_values('percentage', ascending=False, inplace=True)
    print(null.head())

    data = data.drop(['id'], axis=1)

    '''correlation = data.corr()
    upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
    redundant = [col for col in upper if np.any(np.abs(upper[col]) >= 0.95)]
    print(redundant)'''

    neg, pos = np.bincount(data.target.to_numpy())
    total = neg + pos
    print(total)

    train = data[0: int(len(data) * 0.7)]
    val = data[int(len(data) * 0.7): int(len(data) * 0.9)]
    test = data[int(len(data) * 0.9):]

    print(train.shape)

    train_label = train['target']
    train = train.drop(['target'], axis=1)

    val_label = val['target']
    val = val.drop(['target'], axis=1)

    test_label = test['target']
    test = test.drop(['target'], axis=1)

    # Standardize
    mean = train.mean()
    std = train.std() + 1e-12

    train = (train - mean) / std
    val = (val - mean) / std
    test = (test - mean) / std


    def train_loader(data_points, labels, batch_size=32, buffer=1000):
        autotune = tf.data.experimental.AUTOTUNE
        dataset = tf.data.Dataset.from_tensor_slices((data_points, labels))
        dataset = dataset.cache().shuffle(buffer).batch(batch_size, num_parallel_calls=autotune)
        return dataset.prefetch(autotune)


    def val_loader(data_points, labels, batch_size=32):
        autotune = tf.data.experimental.AUTOTUNE
        dataset = tf.data.Dataset.from_tensor_slices((data_points, labels))
        dataset = dataset.cache().batch(batch_size, num_parallel_calls=autotune)
        return dataset.prefetch(autotune)


    def test_loader(data_points, batch_size=32):
        autotune = tf.data.experimental.AUTOTUNE
        dataset = tf.data.Dataset.from_tensor_slices(data_points)
        dataset = dataset.cache().batch(batch_size, num_parallel_calls=autotune)
        return dataset.prefetch(autotune)


    trainloader = train_loader(train.to_numpy(), train_label.to_numpy())
    valloader = val_loader(val.to_numpy(), val_label.to_numpy())
    testloader = test_loader(test.to_numpy())

    for instance, labels in trainloader.take(1):
        print(instance)
        print(print(labels))

    METRICS = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.AUC(name='prc', curve='PR')
    ]

    bias = np.log(pos / neg)

    w0 = (1 / neg) * (total / 2.0)
    w1 = (1 / pos) * (total / 2.0)

    class_weights = {0: w0, 1: w1}
    print(class_weights)
    print(bias)


    def return_model(features, use_bias=None):
        output_bias = None
        if use_bias is not None:
            output_bias = tf.keras.initializers.Constant(use_bias)
        classes = 1
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=features),
            tf.keras.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
            tf.keras.layers.Dense(classes, activation='sigmoid', bias_initializer=output_bias)
        ])
        return model

    model = return_model(features=train.shape[1:], use_bias=bias)
    print(model.summary())

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  metrics=METRICS)

    earlystop = tf.keras.callbacks.EarlyStopping(patience=15, monitor='val_prc')
    checkpoint = tf.keras.callbacks.ModelCheckpoint('best_cancer_model/', save_best_only=True)

    history = model.fit(trainloader, validation_data=valloader, epochs=100, class_weight=class_weights,
                        callbacks=[earlystop, checkpoint])

    model = tf.keras.models.load_model('best_cancer_model/')

    prediction = model.predict(testloader)
    prediction = (prediction >= 0.5).astype(int)

    print(tf.keras.metrics.binary_accuracy(test_label.to_numpy(), tf.squeeze(prediction, axis=-1)))
