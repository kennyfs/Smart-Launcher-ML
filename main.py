import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Concatenate, Dropout
import random


def readcsv(filename):
    return pd.read_csv(filename)


def string_to_dict(input_string_array):
    unique_elements = []
    for element in input_string_array:
        if element not in unique_elements:
            unique_elements.append(element)
    return {element: i for i, element in enumerate(unique_elements)}


def get_dataset_row(
    index,
    app_ids,
    array,
    num_apps,
    length=20,
):
    data1 = []
    data2 = []
    data3 = []
    data4 = []
    target = []
    for i in range(num_apps):
        data1.append(app_ids[max(0, index - length) : index])
        data2.append(array[max(0, index - length) : index])
        data3.append(array[index])
        data4.append(i)
        target.append(1 if i == app_ids[index] else 0)
    return (
        [np.array(x) for x in (data1, data2, data3, data4)],
        np.array(target, dtype=np.float32),
    )


def create_training_dataset(
    app_ids,
    array,
    indexes,
    app_items,
    data_weights,
    app_weights,
    length=20,
    negative_sampling_ratio=0.1,
    max_negative_sampling=5,
):
    data1 = []
    data2 = []
    data3 = []
    data4 = []
    target = []
    sample_weight = []

    negative_samples = max(
        0, min(max_negative_sampling, int(negative_sampling_ratio * app_items))
    )
    for i in indexes:
        app_weight = app_weights[app_ids[i]]
        if i - length < 0:
            continue
        data1.append(app_ids[i - length : i])
        data2.append(array[i - length : i])
        data3.append(array[i])
        data4.append(app_ids[i])
        target.append(1)
        sample_weight.append(data_weights[i] * float(negative_samples) * app_weight)

        # Negative sampling
        for _ in range(negative_samples):
            random_app = np.random.randint(app_items)
            while random_app == app_ids[i]:
                random_app = np.random.randint(app_items)
            data1.append(app_ids[i - length : i])
            data2.append(array[i - length : i])
            data3.append(array[i])
            data4.append(random_app)
            target.append(0)
            sample_weight.append(data_weights[i] * app_weight)

    return (
        [np.array(x) for x in (data1, data2, data3, data4)],
        np.array(target, dtype=np.float32),
        np.array(sample_weight, dtype=np.float32),
    )


def load_data(file):
    df = readcsv(file)
    df = df.iloc[:, 1:]  # Remove ID column

    array = df.to_numpy()
    app_ids = array[:, 1]
    dict = string_to_dict(app_ids)
    app_ids = np.array([dict[i] for i in app_ids])
    app_ids = app_ids.astype(np.int32)

    array = np.delete(array, 1, 1)  # Remove app_id column
    array[:, 0] = array[:, 0] / 24
    array = np.where(
        array == True, 1, array
    )  # 2~6th columns are boolean, so change them to 0 or 1
    array = np.where(array == False, 0, array)

    array[:, 6] = array[:, 6] / np.max(array[:, 6])
    array = array.astype(np.float32)
    return app_ids, array, dict


def train_model(data):

    input_data, target, sample_weight = data

    # start define model with functional API
    x1 = Input(shape=(None,))
    x2 = Input(shape=(None, 7))
    x3 = Input(shape=(7,))
    x4 = Input(shape=())
    embedding_layer = Embedding(300, 16)
    embedding = embedding_layer(x1)
    concat1 = Concatenate()([embedding, x2])
    concat1 = Dense(32, activation="relu")(concat1)
    concat1 = Dropout(0.3)(concat1)
    lstm = LSTM(64, recurrent_dropout=0.3)(concat1)
    current_data_dense = Dense(8, activation="relu")(x3)
    target_embedding = embedding_layer(x4)
    target_embedding = Dense(24, activation="relu")(target_embedding)
    x = Concatenate()([lstm, current_data_dense, target_embedding])
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(16, activation="relu")(x)
    output = Dense(1, activation="sigmoid")(x)
    model = tf.keras.models.Model(inputs=[x1, x2, x3, x4], outputs=output)

    print(model.summary())
    model.compile(
        optimizer="adam", loss="binary_crossentropy", weighted_metrics=["accuracy"]
    )
    print("\nfit:")
    print(
        input_data[0].shape,
        input_data[1].shape,
        input_data[2].shape,
        input_data[3].shape,
        target.shape,
    )
    model.fit(input_data, target, sample_weight=sample_weight, epochs=40)
    return model
    """
    # test
    data, target, sample_weight = create_dataset(test_data, len(dict))
    print("\nevaluate:")
    model.evaluate(data, target, sample_weight=sample_weight)

    # print
    output = model.predict(data)
    answer = target
    for i in range(20):
        print(output[i], answer[i], sample_weight[i])
    """


def get_app_frequency(app_ids, array, app_dict):
    data_length = len(array)
    num_items = len(app_dict)
    app_frequency = np.zeros(num_items)
    for i in range(data_length):
        app_frequency[app_ids[i]] += 1
    app_frequency = app_frequency / np.sum(app_frequency)
    return app_frequency


def save_model(model, filename):
    model.save(filename)


def load_model(filename):
    return tf.keras.models.load_model(filename)


if __name__ == "__main__":
    app_ids, array, dict = load_data("usage database kennyfs.txt")
    print(dict)
    reversed_dict = {v: k for k, v in dict.items()}

    data_length = len(array)
    num_apps = len(dict)
    # Randomly choose indexes for training and test data
    indexes = list(range(data_length))
    random.shuffle(indexes)
    training_indexes = indexes[: int(data_length * 0.8)]
    test_indexes = indexes[int(data_length * 0.8) :]

    print(f"Data rows: {data_length}\nNumber of apps: {num_apps}")

    app_frequency = get_app_frequency(app_ids, array, dict)
    # Print app by frequency
    tmp = list(zip(range(num_apps), app_frequency))
    print(tmp)
    tmp.sort(key=lambda x: x[1], reverse=True)
    id_to_rank = {tmp[i][0]: i for i in range(num_apps)}
    for i, (app, frequency) in enumerate(tmp):
        print(f"{i+1}/{num_apps}", reversed_dict[app], f"{frequency*100:.2f}%")

    app_weights = [app_frequency[i] ** (-0.7) for i in range(num_apps)]
    e = 2.718281828459045
    data_weights = [e ** (i / data_length - 1) for i in range(data_length)]
    training_data = create_training_dataset(
        app_ids,
        array,
        training_indexes,
        num_apps,
        data_weights,
        app_weights,
        length=20,
        negative_sampling_ratio=0.2,
        max_negative_sampling=20,
    )
    train = False
    if train:
        model = train_model(training_data)
        save_model(model, "model.h5")
    else:
        model = load_model("model.h5")

    print(model.summary())

    test_data_app_ids = [app_ids[i] for i in test_indexes]
    low_frequency_apps = []
    low_frequency_indexes = []
    for app_id in range(num_apps):
        frequency = app_frequency[app_id]
        if 0.01 < frequency < 0.05:
            if app_id in test_data_app_ids:
                print(reversed_dict[app_id], f"{frequency*100:.2f}%")
                low_frequency_indexes.extend(
                    [
                        i
                        for i, app in enumerate(test_data_app_ids)
                        if i >= 5 and app == app_id
                    ]
                )
    """
    print(low_frequency_indexes)
    for i in low_frequency_indexes:
        print(reversed_dict[test_data_app_ids[i]])
    """
    for index in low_frequency_indexes:
        print(f"App: {index}, {reversed_dict[test_data_app_ids[index]]}")
        data, target = get_dataset_row(index, app_ids, array, num_apps)
        result = model.predict(data)
        result /= np.sum(result)
        # create a ranking of the results
        tmp = list(zip(result, range(num_apps)))
        tmp.sort(key=lambda x: x[0], reverse=True)
        """
        print("Ranking")
        for i in range(len(tmp)):
            print(f"{reversed_dict[tmp[i][1]]}: {tmp[i][0][0]}, {target[tmp[i][1]]}")
        print()
        """
        # find test_data_app_ids[i] in result, print the ranking and result
        for i in range(len(tmp)):
            if tmp[i][1] == test_data_app_ids[index]:
                print(
                    f"ranking of the answer: {i+1}/{num_apps}, {id_to_rank[test_data_app_ids[index]]}/{num_apps}, result: {tmp[i][0][0]*100:.2f}%"
                )
                break
