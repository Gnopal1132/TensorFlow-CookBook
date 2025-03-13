import string

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def filter_train(textline):
    text = tf.strings.split(textline, sep=',', maxsplit=4)
    cat = text[1]  # train, test
    sentiment = text[2]  # pos, neg, unsup
    return True if cat == 'train' and sentiment != 'unsup' else False


def filter_test(textline):
    text = tf.strings.split(textline, sep=',', maxsplit=4)
    cat = text[1]  # train, test
    sentiment = text[2]  # pos, neg, unsup
    return True if cat == 'test' and sentiment != 'unsup' else False


if __name__ == '__main__':
    path = os.path.join(os.curdir, 'Dataset', "IMDB Dataset", 'imdb.csv')
    dataset = tf.data.TextLineDataset(path)

    """
    Suppose you have several text or csv files files = ["file1.csv", "file2.csv",...] you can read them all in once as
    a lego_dataset like lego_dataset = tf.data.TextLineDataset(files)  as simple as that!! But suppose you want to do preprocessing
    on each files seperately and later want to merge them all. You could do something like:
    dataset1 = tf.data.TextLineDataset("file1.csv").map(preprcess1)
    dataset2 = tf.data.TextLineDataset("file2.csv").map(preprcess2)
    dataset3 = tf.data.TextLineDataset("file3.csv").map(preprcess3)
    
    lego_dataset = dataset1.concatenate(dataset2).concatenate(dataset3)
    
    for line lego_dataset:
        print(line)
        
    """

    # The first line is the column names, so we are ignoring it!, hence the skip
    '''for line in lego_dataset.skip(1).take(5):
        print(line, end='\n')  # It will print the rows as byte string
        print(tf.strings.split(line, sep=',', maxsplit=4), end='\n')  # 4 because its '0,test,neg,0_2.txt,"Once again...
        print('\n\n')'''

    ds_train = dataset.filter(filter_train)  # filter is like map function, but it selects the elements based on a
    # predictor that returns True or False for each element.
    ds_test = dataset.filter(filter_test)

    '''for line in ds_train.skip(1).take(5):
        print(line, end='\n')'''


    # TODO:
    # 1. Set a Vocabulary Size. And clean the text strings
    # 2. Numerical text str to indices. We will use Tokenizer for it.
    # 3. Pad the batches so that we can send it into RNN

    def clean_strings(textline):
        text = tf.strings.split(textline, sep=',', maxsplit=4)
        review = tf.strings.lower(text[4])
        sentiment = tf.strings.lower(text[2])
        refined = tf.strings.regex_replace(review, '<br />', ' ')
        refined = tf.strings.regex_replace(refined, "[{}]".format(string.punctuation), ' ')
        refined = tf.strings.strip(refined)
        return tf.strings.join([sentiment, refined], separator=',')


    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds_train = ds_train.map(clean_strings, num_parallel_calls=AUTOTUNE)
    ds_test = ds_test.map(clean_strings, num_parallel_calls=AUTOTUNE)

    # The maximum length of string
    MAX_SEQUENCE = 250
    VOCAB_SIZE = 10000

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode='int',
        output_sequence_length=MAX_SEQUENCE
    )

    # Note: It's important to only use your training data when calling adapt (using the test set would leak
    # information).

    review = ds_train.map(lambda instance: tf.strings.split(instance, sep=',', maxsplit=1)[1])
    vectorizer.adapt(review)

    # print(vectorizer.get_vocabulary())

    def vectorize_text(text, sentiment):
        text = tf.expand_dims(text, axis=-1)
        sentiment = tf.constant(0, shape=(), dtype=tf.int32) if sentiment == 'neg' \
            else tf.constant(1, shape=(), dtype=tf.int32)
        vectorized = vectorizer(text)  # Note this adds an extra dimension. The output should be (250, ) but this
        # returns (1, 250) so reducing below.
        return tf.squeeze(vectorized, axis=0), sentiment


    '''for line in ds_train.take(1):
        label, text = tf.strings.split(line, sep=',', maxsplit=1)
        vector, _ = vectorize_text(text, label)
        print(vector, end='\n\n')
        print(label)'''

    # Separating the text and the labels
    def separate_text_label(textline):
        split = tf.strings.split(textline, sep=',', maxsplit=1)
        return split[1], split[0]  # Note the second component is the review the first one is just the label


    ds_train = ds_train.map(separate_text_label, num_parallel_calls=AUTOTUNE)
    ds_test = ds_test.map(separate_text_label, num_parallel_calls=AUTOTUNE)

    ds_train = ds_train.map(vectorize_text, num_parallel_calls=AUTOTUNE)
    ds_test = ds_test.map(vectorize_text, num_parallel_calls=AUTOTUNE)

    '''for X, Y in ds_train.take(1):
        print(X.shape)
        print(Y.shape)'''

    # Loading the Dataset
    BATCH_SIZE = 64
    ds_train = ds_train.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    ds_test = ds_test.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)

    for line, label in ds_train.take(1):
        print(line.shape)
        print(label.shape)

    """
    Masking: Now that all samples have a uniform length, the model must be informed that some part of the data is 
    actually padding and should be ignored. That mechanism is masking.

    There are three ways to introduce input masks in Keras models:

    1. Add a keras.layers.Masking layer.
    2. Configure a keras.layers.Embedding layer with mask_zero=True.
    3. Pass a mask argument manually when calling layers that support this argument (e.g. RNN layers).
    
    When using the Functional API or the Sequential API, a mask generated by an Embedding or Masking layer will be 
    propagated through the network for any layer that is capable of using them (for example, RNN layers). Keras will 
    automatically fetch the mask corresponding to an input and pass it to any layer that knows how to use it. 

    For instance, in the following Sequential model, the LSTM layer will automatically receive a mask, which means it 
    will ignore padded values """

    # Defining the Model
    embedding_dim = 64
    model = tf.keras.Sequential([
        # or ues tf.keras.layers.Masking() and make mask_zero=False in Embedding
        tf.keras.layers.Embedding(input_dim=VOCAB_SIZE,
                                  output_dim=embedding_dim,
                                  mask_zero=True,
                                  input_length=MAX_SEQUENCE),
        # Vocab-size + 1, as 0 is added as padding so one extra, if masking is not used else VOCAB_SIZE. Note it
        # outputs: (Batch, MaxSequenceLength, Embedding) Input dimension = ((Batch, MaxSequenceLength)
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)])
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    model.fit(ds_train, epochs=10)
    model.evaluate(ds_test)
