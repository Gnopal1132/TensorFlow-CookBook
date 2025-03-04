import tensorflow as tf
import json
import string
import numpy as np
import pandas as pd
import os
import re

from matplotlib import pyplot as plt


def pad_sentences(sentence):
    sentence = re.sub(f"([{string.punctuation}])", r" \1 ", sentence)
    sentence = re.sub(" +", " ", sentence)
    return sentence.strip()


# Create a TextGenerator checkpoint
class TextGenerator(tf.keras.callbacks.Callback):
    def __init__(self, index_to_word, top_k=10):
        self.index_to_word = index_to_word
        self.word_to_index = {
            word: index for index, word in enumerate(index_to_word)
        }  # <1>

    def sample_from(self, probs, temperature):  # <2>
        probs = probs ** (1 / temperature)
        probs = probs / np.sum(probs)
        return np.random.choice(len(probs), p=probs), probs

    def generate(self, start_prompt, max_tokens, temperature):
        start_tokens = [
            self.word_to_index.get(x, 1) for x in start_prompt.split()
        ]  # <3>
        sample_token = None
        info = []
        while len(start_tokens) < max_tokens and sample_token != 0:  # <4>
            x = np.array([start_tokens])
            y = self.model.predict(x, verbose=0)  # The dimension of x = [batch_size, sequence_length]
            sample_token, probs = self.sample_from(y[0][-1], temperature)
            # Note: y[0][-1] will be the probability distribution over the vocabulary words for the last word.
            info.append({"prompt": start_prompt, "word_probs": probs})
            start_tokens.append(sample_token)
            start_prompt = start_prompt + " " + self.index_to_word[sample_token]
            # We then merge to the start prompt to process for the next one.
        print(f"\ngenerated text:\n{start_prompt}\n")
        return info

    def on_epoch_end(self, epoch, logs=None):
        self.generate("recipe for", max_tokens=100, temperature=1.0)


if __name__ == '__main__':
    dataset_path = os.path.join(os.curdir, 'dataset', 'epirecipes', 'full_format_recipes.json')
    with open(dataset_path) as json_data:
        recipe_data = json.load(json_data)

    # Filter the dataset
    filtered_data = [
        "Recipe for " + x["title"] + " | " + " ".join(x["directions"])
        for x in recipe_data
        if "title" in x
        and x["title"] is not None
        and "directions" in x
        and x["directions"] is not None
    ]

    # In order to make the data ready, we need to make few adjustments first
    # Step 1: Tokenization: Every word will be considered as a token, we won't ignore punctuation marks, because we also
    # want the model to understand when to put a comma or full step. So we will first put spaces before and after a
    # punctuation mark so that it can be considered a valid token.
    filtered_data = [pad_sentences(instance) for instance in filtered_data]

    # Step 2: Creating a tf.dataset
    BATCH_SIZE = 64
    VOCAB_SIZE = 10000
    MAX_LEN = 200
    EMBEDDING_DIM = 100
    N_UNITS = 128

    dataset = tf.data.Dataset.from_tensor_slices(filtered_data)
    dataset = dataset.batch(BATCH_SIZE).shuffle(1000).prefetch(tf.data.AUTOTUNE)

    for instance in dataset.take(1):
        print(instance)
        print(instance.shape)

    # Step 3: Create a Vectorization layer
    # A preprocessing layer which maps text features to integer sequences.
    vectorize_layer = tf.keras.layers.TextVectorization(
        standardize='lower',
        output_mode='int',
        max_tokens=VOCAB_SIZE,
        output_sequence_length=MAX_LEN + 1
    )
    vectorize_layer.adapt(dataset)
    vocabulary = vectorize_layer.get_vocabulary()
    text_generation = TextGenerator(vocabulary)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(os.curdir, 'checkpoints', 'lstm_model.keras'),
        save_freq="epoch",
        verbose=0,
        save_best_only=True
    )

    def prepare_dataset(sentence):
        x = tf.expand_dims(sentence, axis=-1)
        tokenized_sentence = vectorize_layer(x)
        x = tokenized_sentence[:, :-1]
        y = tokenized_sentence[:, 1:]
        return x, y


    train_dataset = dataset.map(prepare_dataset)
    for x, y in train_dataset.take(1):
        print(x, y)

    # inputs = tf.keras.layers.Input(shape=(None,), dtype="int32")
    # x = tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM)(inputs)
    # x = tf.keras.layers.LSTM(N_UNITS, return_sequences=True)(x)
    # outputs = tf.keras.layers.Dense(VOCAB_SIZE, activation="softmax")(x)
    # lstm = tf.keras.Model(inputs, outputs)
    # print(lstm.summary())
#
    # lstm.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='adam')
    # history = lstm.fit(train_dataset, epochs=25, callbacks=[model_checkpoint_callback, text_generation])
    # pd.DataFrame(history.history).plot(figsize=(10, 5))
    # plt.grid(True)
    # plt.show()
