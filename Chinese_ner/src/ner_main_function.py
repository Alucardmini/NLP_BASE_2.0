#!/usr/bin/python
#coding:utf-8
'''
install keras-contrib
    pip3 install git+https://www.github.com/keras-team/keras-contrib.git
'''


from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Embedding, Dropout, TimeDistributed, Input
from keras_contrib.layers.crf import CRF
# from keras_contrib.utils import save_load_utils
from Chinese_ner.src.ner_datas import ner_datas
from keras.preprocessing import sequence


if __name__ == "__main__":

    model_path = '../data/bilstm_crf_4_ner.h5'

    data_loader = ner_datas()
    # 载入数据
    (train_vocabs, train_labels), (test_vocabs, test_labels) = data_loader.load_data()

    vocab_size = 4768
    embedding_size = 128
    time_stamps = 100
    hidden_units = 200
    dropout_rate = 0.3
    num_class = 7
    max_len = 100
    train_vocabs = sequence.pad_sequences(train_vocabs, max_len)
    test_vocabs = sequence.pad_sequences(test_vocabs, max_len)

    train_labels = sequence.pad_sequences(train_labels, max_len)
    test_labels = sequence.pad_sequences(test_labels, max_len)
    train_labels = train_labels.reshape(train_labels.shape[0], train_labels.shape[-1], 1)
    test_labels = test_labels.reshape(test_labels.shape[0], test_labels.shape[-1], 1)
    print( train_vocabs.shape)
    print( train_labels.shape)

    model = Sequential()
    # model.add(Input(shape=(None,), dtype='int32'))
    model.add(Embedding(vocab_size, embedding_size, mask_zero=True))
    model.add(Bidirectional(LSTM(hidden_units, return_sequences=True)))
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(LSTM(hidden_units, return_sequences=True)))
    model.add(Dropout(dropout_rate))
    # model.add(TimeDistributed(Dense(num_class)))
    model.add(TimeDistributed(Dense(num_class)))
    model.compile(
        optimizer='rmsprop',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # crf_layer = CRF(num_class)
    # # model.add(crf_layer)
    # model.add(TimeDistributed(crf_layer))
    # model.compile(
    #     optimizer='rmsprop',
    #     loss=crf_layer.loss_function,
    #     metrics=[crf_layer.accuracy]
    # )


    print(model.summary())
    # model.fit(train_vocabs, train_labels, batch_size=128, epochs=1)
    # model.save(model_path)

    test_content = "北京到上海多少公里"
    pred_x = data_loader.vector_sent_only(list(test_content))

    y = model.predict(pred_x)
    print(y.shape)
    print(y[1])
    # print(y)


