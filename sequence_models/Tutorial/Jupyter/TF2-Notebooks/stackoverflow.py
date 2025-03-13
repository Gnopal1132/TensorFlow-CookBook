import tensorflow as tf
import os
import string

if __name__ == '__main__':
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz'
    dataset = tf.keras.utils.get_file(fname='stack_overflow_16k',
                                      origin=url,
                                      untar=True, cache_dir='.',
                                      cache_subdir=os.path.join(os.curdir, 'lego_dataset', 'stackoverflow'))
    path = os.path.join(os.curdir, 'lego_dataset', 'stackoverflow')

    BATCH = 32
    SEED = 123
    train = tf.keras.utils.text_dataset_from_directory(
        os.path.join(path, 'train'),
        subset='training',
        validation_split=0.2,
        seed=SEED,
        batch_size=BATCH
    )
    val = tf.keras.utils.text_dataset_from_directory(
        os.path.join(path, 'train'),
        subset='validation',
        validation_split=0.2,
        seed=SEED,
        batch_size=BATCH
    )
    test = tf.keras.utils.text_dataset_from_directory(
        os.path.join(path, 'test'),
        batch_size=BATCH
    )

    '''for text, label in train.take(1):
        print(text.shape)
        print(label.shape)
        print(text)
        print(label)'''

    def clean_string(instance):
        instance = tf.strings.lower(instance)
        instance = tf.strings.regex_replace(instance, '[{}]'.format(string.punctuation), '')
        instance = tf.strings.strip(instance)
        return instance


    MAX_SEQUENCE = 250
    VOCABSIZE = 10000

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=VOCABSIZE,
        standardize=clean_string,
        output_mode='int',
        output_sequence_length=MAX_SEQUENCE
    )
    data = train.map(lambda text, label: text, num_parallel_calls=tf.data.AUTOTUNE)
    vectorizer.adapt(data)

    def vectorize(text, label):
        return vectorizer(text), label

    train = train.map(vectorize, num_parallel_calls=tf.data.AUTOTUNE)
    val = val.map(vectorize, num_parallel_calls=tf.data.AUTOTUNE)
    test = test.map(vectorize, num_parallel_calls=tf.data.AUTOTUNE)

    for text, label in train.take(1):
        print(text.shape)
        print(label.shape)
        print(text)
        print(label)

    train = train.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
    val = val.cache().prefetch(tf.data.AUTOTUNE)
    test = test.cache().prefetch(tf.data.AUTOTUNE)

    # Defining the Model
    embedding_dim = 64
    l2 = tf.keras.regularizers.l2(0.01)
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=VOCABSIZE,
                                  output_dim=embedding_dim,
                                  mask_zero=True,
                                  input_length=MAX_SEQUENCE),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(50, return_sequences=True, kernel_regularizer=l2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(50, kernel_regularizer=l2),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4, activation='softmax')])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer='adam',
                  metrics=['accuracy'])

    earlystop = tf.keras.callbacks.EarlyStopping(patience=15)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('best_overflow/', save_best_only=True)

    history = model.fit(train, validation_data=val, epochs=150, callbacks=[earlystop, checkpoint])

    model = tf.keras.models.load_model('best_overflow/')
