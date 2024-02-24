import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Concatenate,Lambda


def readcsv(filename):
    return pd.read_csv(filename)


def string_to_dict(input_string_array):
    unique_elements = []
    for element in input_string_array:
        if element not in unique_elements:
            unique_elements.append(element)
    return {element: i for i, element in enumerate(unique_elements)}


def create_dataset(data, app_items, length=10):
    app_ids, array = data
    data1 = []
    data2 = []
    data3 = []
    data4 = []
    target = []

    for i in range(len(app_ids) - length):
        data1.append(app_ids[i : i + length])
        data2.append(array[i : i + length])
        data3.append(array[i + length])
        data4.append(app_ids[i + length])
        target.append(1)

        # Negative sampling
        for _ in range(2):
            random_app = np.random.randint(app_items)
            while random_app == app_ids[i + length]:
                random_app = np.random.randint(app_items)
            data1.append(app_ids[i : i + length])
            data2.append(array[i : i + length])
            data3.append(array[i + length])
            data4.append(random_app)
            target.append(0)

    return [np.array(x) for x in (data1, data2, data3, data4)], np.array(
        target, dtype=np.float32
    )




if __name__ == "__main__":
    df = readcsv("usage database.txt")
    df = df.iloc[:, 1:]  # Remove ID column

    array = df.to_numpy()
    app_ids = array[:, 1]
    dict = string_to_dict(app_ids)
    app_ids = np.array([dict[i] for i in app_ids])
    app_ids = app_ids.astype(np.int32)
    print(app_ids)

    array = np.delete(array, 1, 1)  # Remove app_id column
    array[:, 0] = array[:, 0] / 24
    array = np.where(
        array == True, 1, array
    )  # 2~6th columns are boolean, so change them to 0 or 1
    array = np.where(array == False, 0, array)

    array[:, 6] = array[:, 6] / np.max(array[:, 6])
    array = array.astype(np.float32)
    print("array exsample", array[0].tolist())
    
    data_length =len(array)
    training_length = int(data_length * 0.6)
    training_data = app_ids[:training_length],array[:training_length]
    test_data = app_ids[training_length:],array[training_length:]


    # start define model with functional API
    x1= Input(shape=(10,))
    x2= Input(shape=(10,7))
    x3= Input(shape=(7,))
    x4= Input(shape=())
    embedding_layer = Embedding(300, 16)
    embedding = embedding_layer(x1)
    concat1 = Concatenate()([embedding, x2])
    lstm = LSTM(8)(concat1)
    current_data_dense = Dense(8, activation='relu')(x3)
    target_embedding = embedding_layer(x4)
    concat2 = Concatenate()([lstm, current_data_dense, target_embedding])
    output = Dense(1, activation='sigmoid')(concat2)
    model = tf.keras.models.Model(inputs=[x1, x2, x3, x4], outputs=output)

    print(model.summary())
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    data, target = create_dataset(training_data, len(dict), length=10)
    print(data[0].shape, data[1].shape, data[2].shape, data[3].shape, target.shape)
    model.fit(data, target, epochs=30)

    # test
    data, target = create_dataset(test_data, len(dict), length=10)
    print("evaluate")
    model.evaluate(data, target)

    # print
    output = model.predict(data)
    answer = target
    for i in range(20):
        print(output[i], answer[i])
