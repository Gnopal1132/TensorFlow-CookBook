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

    train = train.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
    val = val.cache().prefetch(tf.data.AUTOTUNE)
    test = test.cache().prefetch(tf.data.AUTOTUNE)

    for text, label in train.take(1):
        print(text.shape)
        print(label.shape)
