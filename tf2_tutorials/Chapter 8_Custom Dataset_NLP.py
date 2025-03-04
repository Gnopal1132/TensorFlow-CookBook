import tensorflow as tf
import os
import pandas as pd
import spacy

spacy_eng = spacy.load("en_core_web_sm")


class CustomDataset:
    def __init__(self, root_dir, caption_file,
                 frequency_threshold=5, buffer_size=1000,
                 batch_size=32, max_len=120, padding='pre', image_width=224, image_height=224):
        """

        :param root_dir: The root directory of image folder
        :param caption_file: The caption text file, since its space separated, so its CSV.
        :param frequency_threshold: This parameter will give you the number if the word
        is not appearing that many times, we will ignore it.
        """
        self.max_len = max_len
        self.padding = padding
        self.size = (image_width, image_height)
        self.root_dir = root_dir
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.dataframe = pd.read_csv(caption_file)

        # Get image and caption column
        self.img = [os.path.join(self.root_dir, instance) for instance in self.dataframe["image"].tolist()]
        self.caption = self.dataframe["caption"].tolist()

        # Let's first create the Vocabulary
        self.vocab = Vocabulary(frequency_threshold=frequency_threshold)
        self.vocab.create_vocabulary(self.caption)

    def read_image(self, image, caption):
        image = tf.io.read_file(image)
        image = tf.image.decode_jpeg(image)
        image = tf.image.resize(image, size=self.size)
        image = tf.cast(image, dtype=tf.float32)
        return image, caption

    def return_loader(self, train_size=0.8):
        AUTOTUNE = tf.data.AUTOTUNE
        train_length = int(train_size * len(self.img))
        dataset = tf.data.Dataset.from_tensor_slices((self.img, self.caption))
        dataset = dataset.shuffle(self.buffer_size)

        # Splitting into lego_dataset
        train = dataset.take(train_length)
        validation = dataset.skip(train_length)

        train = train.map(self.read_image, num_parallel_calls=AUTOTUNE)
        validation = validation.map(self.read_image, num_parallel_calls=AUTOTUNE)

        train = train.map(lambda image, caption: (image, tf.py_function(func=self.text_to_sequence,
                                                                        inp=[caption],
                                                                        Tout=tf.float32)),
                          num_parallel_calls=AUTOTUNE)
        validation = validation.map(lambda image, caption: (image, tf.py_function(func=self.text_to_sequence,
                                                                                  inp=[caption],
                                                                                  Tout=tf.float32)),
                                    num_parallel_calls=AUTOTUNE)
        train = train.cache().batch(self.batch_size).prefetch(AUTOTUNE)
        validation = validation.cache().batch(self.batch_size).prefetch(AUTOTUNE)

        return train, validation

    def text_to_sequence(self, text):
        text = text.numpy().decode('UTF-8')
        numerical_caption = [self.vocab.stoi["<SOS>"]]
        numerical_caption += self.vocab.numericalize(text)
        numerical_caption.append(self.vocab.stoi["<EOS>"])

        # Note, pad_sequence expects a sequence i.e. [[1]] in such format
        numerical_caption = tf.keras.utils.pad_sequences([numerical_caption],
                                                         maxlen=self.max_len,
                                                         padding=self.padding,
                                                         value=self.vocab.stoi["<PAD>"])
        return tf.convert_to_tensor(numerical_caption, dtype=tf.float32)


class Vocabulary:
    def __init__(self, frequency_threshold):
        # index to string, pad, start, end, unknown
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = frequency_threshold

    def tokenize(self, string):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(string)]

    def create_vocabulary(self, captions):
        frequency = {}
        index = 4  # because till 3 we have reserved words
        for sentence in captions:
            for word in self.tokenize(sentence):
                frequency[word] = 1 + frequency.get(word, 0)

                if frequency[word] == self.freq_threshold:
                    self.stoi[word] = index
                    self.itos[index] = word
                    index += 1

    def numericalize(self, text):
        tokenized_text = self.tokenize(text)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]


if __name__ == '__main__':
    image_path = os.path.join(os.curdir, 'Flickr Dataset', 'Images')
    caption_path = os.path.join(os.curdir, 'Flickr Dataset', 'captions.txt')

    custom_dataset = CustomDataset(root_dir=image_path, caption_file=caption_path,
                                   max_len=120, padding='post')
    train, validation = custom_dataset.return_loader()
    for image, caption in train.take(1):
        print(image.shape)
        print(caption.shape)
