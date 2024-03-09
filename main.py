import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Concatenate, Dropout
import random
import matplotlib.pyplot as plt


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
    app_frequency,
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

        # Random negative sampling
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

        # Popularity-based negative sampling(Seems useless or there are some bugs)

        # other_apps = [i for i in range(app_items) if i != app_ids[i]]
        # u = np.random.random((app_items))
        # k = [0.0] * app_items
        # for i in range(app_items):
        #     if i == app_ids[i]:
        #         continue
        #     k[i] = u[i] ** (1 / app_frequency[i])
        # for app in range(app_items):
        #     print(f'app: {app}, frequency: {app_frequency[app]}, k: {k[app]}')
        # other_apps.sort(key=lambda x: k[x], reverse=True)
        # for negative_app in range(negative_samples):
        #     print(f'negative app: {negative_app}, frequency: {app_frequency[negative_app]}')
        #     data1.append(app_ids[i - length : i])
        #     data2.append(array[i - length : i])
        #     data3.append(array[i])
        #     data4.append(negative_app)
        #     target.append(0)
        #     sample_weight.append(data_weights[i] * app_weight)
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


def train_model(data, hyper_parameter):

    input_data, target, sample_weight = data

    # start define model with functional API
    x1 = Input(shape=(None,))
    x2 = Input(shape=(None, 7))
    x3 = Input(shape=(7,))
    x4 = Input(shape=())
    embedding_layer = Embedding(100, hyper_parameter["embedding_dim"])
    embedding = embedding_layer(x1)
    concat1 = Concatenate()([embedding, x2])
    concat1 = Dense(32, activation="relu")(concat1)
    concat1 = Dropout(0.5)(concat1)
    lstm = LSTM(64, recurrent_dropout=0.5)(concat1)
    current_data_dense = Dense(8, activation="relu")(x3)
    target_embedding = embedding_layer(x4)
    target_embedding = Dense(8, activation="relu")(target_embedding)
    toConcat = [target_embedding]
    if hyper_parameter["past_data"]:
        toConcat.append(lstm)
    if hyper_parameter["current_data"]:
        toConcat.append(current_data_dense)
    x = Concatenate()(toConcat)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(16, activation="relu")(x)
    output = Dense(1, activation="sigmoid")(x)
    model = tf.keras.models.Model(inputs=[x1, x2, x3, x4], outputs=output)

    print(model.summary())
    model.compile(
        optimizer=hyper_parameter["optimizer"],
        loss=hyper_parameter["loss"],
        weighted_metrics=["accuracy"],
    )
    print("\nfit:")
    print(
        input_data[0].shape,
        input_data[1].shape,
        input_data[2].shape,
        input_data[3].shape,
        target.shape,
    )
    model.fit(
        input_data,
        target,
        sample_weight=sample_weight,
        epochs=hyper_parameter["epochs"],
        batch_size=hyper_parameter["batch_size"],
    )
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


def get_app_frequency(app_ids, app_dict):
    data_length = len(app_ids)
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
    hyper_parameter = {
        "test_id": 0,
        'exclude_social_media': False,
        "length": 20,
        "embedding_dim": 8,
        "past_data": True,
        "current_data": True,
        "max_negative_sampling": 20,
        "low_frequency_factor": -0.7,
        "epochs": 20,
        "batch_size": 64,
        "optimizer": "adam",
        "loss": "mse",
    }

    app_ids, array, dict = load_data("usage database kennyfs.txt")

    print(dict)
    reversed_dict = {v: k for k, v in dict.items()}

    data_length = len(array)
    num_apps = len(dict)
    # Randomly choose indexes for training and test data
    exclude_social_media = hyper_parameter["exclude_social_media"]
    if exclude_social_media:
        indexes = [
            i
            for i in range(data_length)
            if app_ids[i]
            not in (
                dict["com.twitter.android"],
                dict["com.facebook.katana"],
                dict["org.telegram.messenger"],
            )
        ]
    else:
        indexes = list(range(data_length))
    random.seed("Smart-Launcher-ML")
    random.shuffle(indexes)
    training_rows = int(len(indexes) * 0.9)
    training_indexes = indexes[:training_rows]
    test_indexes = indexes[training_rows:]

    print(f"Data rows: {data_length}\nNumber of apps: {num_apps}")

    app_frequency = get_app_frequency([app_ids[i] for i in indexes], dict)
    # Print app by frequency
    tmp = list(zip(range(num_apps), app_frequency))
    print(tmp)
    tmp.sort(key=lambda x: x[1], reverse=True)
    id_to_rank = {tmp[i][0]: i + 1 for i in range(num_apps)}
    for i, (app, frequency) in enumerate(tmp):
        print(f"{i+1}/{num_apps}", reversed_dict[app], f"{frequency*100:.2f}%")

    app_weights = [
        app_frequency[i] ** hyper_parameter["low_frequency_factor"]
        for i in range(num_apps)
    ]
    e = 2.718281828459045
    # data_weights = [e ** ((i / data_length) / 3) for i in range(data_length)]
    data_weights = [1.0 for i in range(data_length)]
    training_data = create_training_dataset(
        app_ids,
        array,
        app_frequency,
        training_indexes,
        num_apps,
        data_weights,
        app_weights,
        length=hyper_parameter["length"],
        negative_sampling_ratio=1.0,
        max_negative_sampling=hyper_parameter["max_negative_sampling"],
    )
    print(training_data[2])
    train = True
    if train:
        model = train_model(training_data, hyper_parameter)
        # save_model(model, "model.h5")
    else:
        model = load_model("model.h5")

    print(model.summary())

    # test
    rankings = [[] for i in range(num_apps)]
    for index in test_indexes:
        print(f"App: {index}, {reversed_dict[app_ids[index]]}")
        data, target = get_dataset_row(index, app_ids, array, num_apps)
        result = model.predict(data)
        tmp = list(zip(result, range(num_apps)))
        tmp.sort(key=lambda x: x[0], reverse=True)
        # find test_data_app_ids[i] in result, print the ranking and result
        for i in range(len(tmp)):
            if tmp[i][1] == app_ids[index]:
                print(f"find at {i+1}")
                rankings[app_ids[index]].append(i + 1)
                break

    dataToDraw = list(zip(range(num_apps), rankings, app_frequency))
    filtered_data = [
        (app, ranking, frequency)
        for app, ranking, frequency in dataToDraw
        if len(ranking) > 0
    ]
    filtered_data.sort(key=lambda x: x[2], reverse=True)
    plot_ids, plot_rankings, plot_frequencies = zip(*filtered_data)
    length = len(plot_ids)
    plt.figure(figsize=(12, 6))
    plt.boxplot(plot_rankings)
    for i, id in enumerate(plot_ids):
        plt.scatter(i + 1, id_to_rank[id], color="red")
    plt.xticks(
        range(1, length + 1),
        [reversed_dict[i] for i in plot_ids],
        rotation=90,
    )
    plt.xlabel("App Names")
    plt.ylabel("Ranking")
    plt.title("Ranking Distribution of Apps")
    test_id = hyper_parameter["test_id"]
    with open(f"hyper_parameter_{test_id}.txt", "w") as f:
        for key, value in hyper_parameter.items():
            if key != "test_id":
                f.write(f"{key}: {value}\n")
    plt.savefig(f"ranking_distribution_{test_id}.png")
    plt.show()
    # test low frequency apps
    """
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

    for index in low_frequency_indexes:
        print(f"App: {index}, {reversed_dict[test_data_app_ids[index]]}")
        data, target = get_dataset_row(index, app_ids, array, num_apps)
        result = model.predict(data)
        result /= np.sum(result)
        # create a ranking of the results
        tmp = list(zip(result, range(num_apps)))
        tmp.sort(key=lambda x: x[0], reverse=True)
        # find test_data_app_ids[i] in result, print the ranking and result
        for i in range(len(tmp)):
            if tmp[i][1] == test_data_app_ids[index]:
                print(
                    f"ranking of the answer: {i+1}/{num_apps}, {id_to_rank[test_data_app_ids[index]]}/{num_apps}, result: {tmp[i][0][0]*100:.2f}%"
                )
                break
    """
