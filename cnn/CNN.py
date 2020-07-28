import re
import os
import pickle
import numpy as np
import pandas as pd
import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Lambda, Embedding, Conv1D, GlobalMaxPooling1D
from keras.initializers import Constant
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import LabelEncoder


def clean_doc(sentence):
    """
    :param sentence: raw sentence
    :return: cleaned sentence
    """
    # translator = str.maketrans('', '', string.punctuation)
    cleaned = re.sub(r'[^\x00-\x7F]+', ' ', sentence)
    cleaned = " ".join([i for i in cleaned.lower().split()]).encode('UTF-8', 'ignore').decode('ascii', 'ignore')
    cleaned = ''.join([i for i in cleaned if not i.isdigit()])
    # cleaned = str(cleaned).translate(translator)

    return str(cleaned)


def get_word_embedding_matrix():
    """
    :return: word embedding matrix
    """
    print('Indexing word vectors.')
    embeddings_index = {}
    with open(GLOVE_PATH) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))

    return embeddings_index


def get_training_data():
    """
    It's expected that the input csv file has two columns: "text" and "label"
    :return:
    """
    data = pd.read_csv(TRAINING_PATH)
    data['text'] = data['text'].astype(str)
    data = data.drop_duplicates()
    data['text'] = data['text'].apply(lambda x: clean_doc(x))
    mask = (data['text'].str.len() > 5)  # remove short inputs
    data = data.loc[mask]
    data = data.reset_index()

    return data


def get_training_pad_label(training):
    """
    :param training: training data
    :return: padded text and one-hot encodings of labels
    """
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS_JOB)
    tokenizer.fit_on_texts(training['text'])
    sequences = tokenizer.texts_to_sequences(training['text'])

    # sequences = [i for j, i in enumerate(sequences) if j not in empty_index]

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    pad_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH_JOB)

    encoder = LabelEncoder()
    encoder.fit(training['label'])
    encoded_y = encoder.transform(training['label'])
    labels = np_utils.to_categorical(encoded_y)

    total_title = training['label'].nunique()

    # saving job tokenizer
    with open('job_tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return word_index, pad_data, labels, encoder, total_title


def train_test_split_index(pad_data, labels):
    indices = np.arange(pad_data.shape[0])
    np.random.shuffle(indices)
    pad_data = pad_data[indices]
    labels = labels[indices]
    num_validation_samples = int(VALIDATION_SPLIT * pad_data.shape[0])

    return pad_data, labels, num_validation_samples


def split_train_val(num_validation_samples, pad_data, labels):
    job_train = pad_data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    job_val = pad_data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]

    return job_train, job_val, y_train, y_val


def build_job_embedding(word_index, embeddings_index):
    print('Preparing embedding matrix.')

    # prepare embedding matrix
    num_words = min(MAX_NUM_WORDS_JOB, len(word_index)) + 1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NUM_WORDS_JOB:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer (for job description)
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=MAX_SEQUENCE_LENGTH_JOB,
                                trainable=False)

    return embedding_layer


def build_job_layer(embedding_layer, filters=1000, kernel_size=5, strides=1, activation='tanh', out_shape=100):
    jdes_sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH_JOB,), dtype='float32')
    jdes = embedding_layer(jdes_sequence_input)
    jdes = Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, activation=activation)(jdes)
    jdes = GlobalMaxPooling1D()(jdes)
    jdes = Dense(1000, activation='tanh')(jdes)
    jdes = Dropout(0.3)(jdes)
    jd_embedding = Dense(out_shape, activation='relu')(jdes)

    return jdes_sequence_input, jd_embedding

def get_title_embedding_matrix(encoder, embeddings_index, total_title, out_shape=100):
    """
    :param encoder:
    :param embeddings_index:
    :param total_title:
    :param out_shape:
    :return: tensor of title constant embedding matrix
    """
    title_embedding = np.zeros(shape=(total_title, out_shape))
    count = 0
    for title in encoder.classes_:
        emb = np.zeros(out_shape,)
        for i in range(len(title.split())):
            if title.split()[i] in embeddings_index:
                emb += embeddings_index[title.split[i]]
        emb = emb / len(title.split())
        title_embedding[count] = emb
        count += 1
    title_embedding = np.float32(title_embedding)

    return K.variable(title_embedding)


def cosine_distance(jd):
    jd = K.l2_normalize(jd, axis=-1)
    jt_six = K.l2_normalize(title_embedding, axis=-1)

    return K.dot(jd, K.transpose(jt_six))

def get_callbacks():
    filepath = "weights_best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    callbacks_list = [checkpoint, es]

    return callbacks_list


def compile_and_fit_model(callbacks_list, jdes_sequence_input, result, job_train,
                          y_train, job_val, y_val):
    model = Model(inputs=jdes_sequence_input, outputs=result)
    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
    model.fit(job_train, y_train, validation_split=VALIDATION_SPLIT, batch_size=BATCH_SIZE,
              epochs=EPOCHS, callbacks=callbacks_list)

    # model_json = model.to_json()
    # with open("title_prediction_model.json", "w") as json_file:
    #     json_file.write(model_json)
    # model.save_weights("title_prediction_model.h5")
    # print("Saved model to disk")

    model.save("title_prediction_model.h5")

    print("The model should start evaluating now!!")
    validation_results = model.evaluate(job_val, y_val, batch_size=BATCH_SIZE)
    # model.save_weights("title_prediction_model.h5")
    print("Validation loss: " + str(validation_results[0]))
    print("Validation metrics: " + str(validation_results[1]))
	
def build_model_and_print_result():
    embeddings_index = get_word_embedding_matrix()
    training = get_training_data()
    word_index, pad_data, labels, encoder, total_title = get_training_pad_label(training)
    pad_data, labels, num_validation_samples = train_test_split_index(pad_data, labels)
    embedding_layer_job = build_job_embedding(word_index, embeddings_index)
    jdes_sequence_input, jd_embedding = build_job_layer(embedding_layer_job, filters=1000, kernel_size=5, strides=1,
                                                        activation='tanh', out_shape=100)
    global title_embedding
    title_embedding = get_title_embedding_matrix(encoder, embeddings_index, total_title, out_shape=100)
    result = Lambda(cosine_distance)(jd_embedding)

    job_train, job_val, y_train, y_val = split_train_val(num_validation_samples, pad_data, labels)
    callbacks_list = get_callbacks()

    jd_embedding_model = Model(inputs=[jdes_sequence_input], outputs=[jd_embedding])

    compile_and_fit_model(callbacks_list, jdes_sequence_input, result, job_train,
                          y_train, job_val, y_val)

    print("Start saving embedding model!!")
    # # serialize embedding model to JSON
    # model_json = jd_embedding_model.to_json()
    # with open("jd_embedding_model.json", "w") as json_file:
    #     json_file.write(model_json)
    # # serialize weights to HDF5
    # jd_embedding_model.save_weights("jd_embedding_model.h5")
    jd_embedding_model.save("jd_embedding_model.h5")
    print("Saved embedding model to disk")

    return None


if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # set parameters:
    MAX_SEQUENCE_LENGTH_JOB = 250
    # MAX_SEQUENCE_LENGTH_TITLE = 5
    EMBEDDING_DIM = 100
    VALIDATION_SPLIT = 0.2
    MAX_NUM_WORDS_JOB = 40000
    # MAX_NUM_WORDS_TITLE = 2000
    EPOCHS = 1
    LEARNING_RATE = 0.1
    BATCH_SIZE = 64
    LOSS = 'binary_crossentropy'
    OPTIMIZER = 'adam'
    METRICS = ['accuracy']
    GLOVE_PATH = '/Users/ruozheng/Documents/DS/glove.6B.100d.txt'
    TRAINING_PATH = "/Users/ruozheng/Documents/DS/title_work_experience.csv"

    build_model_and_print_result()