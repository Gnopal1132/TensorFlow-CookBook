import tensorflow as tf

if __name__ == '__main__':
    # Examples to learn the concepts, these are my corpus
    sentences = [
        'I love my dog!',
        'I love my cat',
        'You love my dog',
        'Do you think my dog is amazing'
    ]
    print(sentences)
    # Tokenizer, will create dictionary of word encodings,
    # and create a vector out of those sentences.

    # num_words = vocabulary, take 10 most frequently words from corpus
    # THe words not in vocab will be mapped to <oov>
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10, oov_token='<OOV>')
    """
    By default, all punctuation is removed, turning the texts into space-separated sequences of lower case 
    words (words may include the ' character). These sequences are then split into lists of tokens. 
    They will then be indexed or vectorized.
    """
    # Note, default tokenizer will create mapping for all the word irrespective of vocab size.
    # but later when we will use it to create embeddings it will consider only till vocab size
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    print(word_index)

    # The word which is not in vocab will be ignored. Here vocab size will play a
    # vital role. But since we used oov token, the ignored words will be replaced
    # with <oov>. Since sentences can be of varying length. We need to pad them
    sequences = tokenizer.texts_to_sequences(sentences)
    print(sequences)

    # To make them of equal size, we use padding.
    padded_sentence = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=10,
                                                                    padding='pre',
                                                                    truncating='post')
    # maxlen can be set, maybe say u found the max length sentence and set it to that
    # but it might lead to many unnecessary computation. So this is hyperparameter.
    # if sentence is bigger that 10, then truncating will happen, if smaller then padding.

    # This you can pass to the model.
    print(padded_sentence)


