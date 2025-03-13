import tensorflow as tf
import os
import shutil
import string

if __name__ == '__main__':
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    dataset = tf.keras.utils.get_file(fname='aclImdb_v1',
                                      origin=url,
                                      cache_dir='.',
                                      untar=True,
                                      cache_subdir='')
    path = os.path.join(os.curdir, 'aclImdb')
    BATCH = 32
    SEED = 123
    shutil.rmtree(os.path.join(path, 'train', 'unsup'))
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
    for text, label in train.take(1):
        print(text.shape)
        print(label.shape)
        # print(text)
        # print(label)


    def clean_string(instance):
        instance = tf.strings.lower(instance)
        instance = tf.strings.regex_replace(instance, '<br />', '')
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
        # print(text)
        # print(label)

    train = train.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
    val = val.cache().prefetch(tf.data.AUTOTUNE)
    test = test.cache().prefetch(tf.data.AUTOTUNE)

    # Defining the Model
    embedding = 64
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=VOCABSIZE,
                                  output_dim=embedding,
                                  input_length=MAX_SEQUENCE,
                                  mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer='adam',
                  metrics=['accuracy'])

    earlystop = tf.keras.callbacks.EarlyStopping(patience=15)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('best_imdb/', save_best_only=True)

    history = model.fit(train, validation_data=val, epochs=150, callbacks=[earlystop, checkpoint])

    model = tf.keras.models.load_model('best_imdb/')
