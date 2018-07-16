import math
from IPython import display
from matplotlib import  cm
from matplotlib import  gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_data_frame = pd.read_csv("/Users/txg/Downloads/california_housing_train.csv", sep=",")

california_housing_data_frame = california_housing_data_frame.reindex(np.random.permutation
                                                                      (california_housing_data_frame.index))
california_housing_data_frame['median_house_value'] /= 1000.0


# my_features = california_housing_data_frame[["total_rooms"]]
#
# feature_columns = [tf.feature_column.numeric_column("total_rooms")]
#
# my_targets = california_housing_data_frame["median_house_value"]
#
# my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
#
# my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
#
#
# linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns, optimizer=my_optimizer)


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    features = {key: np.array(value) for key, value in dict(features).items()}

    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

#
# _ = linear_regressor.train(
#     input_fn=lambda: my_input_fn(my_features, my_targets),
#     steps=100
# )
#
# prediction_input_fn = lambda: my_input_fn(my_features, my_targets, num_epochs=1, shuffle=False)
# predictions = linear_regressor.predict(input_fn=prediction_input_fn)
#
# predictions = np.array([item['predictions'][0] for item in predictions])
#
# mean_squared_error = metrics.mean_squared_error(predictions, my_targets)
#
# root_mean_squared_error = math.sqrt(mean_squared_error)
#
# print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
#
# print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)
#
# sample = california_housing_data_frame.sample(n=300)
#
# x_0 = sample["total_rooms"].min()
# x_1 = sample["total_rooms"].max()
#
# weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
# bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')
#
# y_0 = weight * x_0 + bias
# y_1 = weight * x_1 + bias
#
# plt.plot([x_0, x_1], [y_0, y_1], c='r')
#
# plt.ylabel("median_house_value")
# plt.xlabel("total_rooms")
#
# plt.scatter(sample["total_rooms"], sample["median_house_value"])
#
# plt.show()


def train_model(learning_rate, steps, batch_size, input_feature="total_rooms"):
    periods = 10
    steps_per_periods = steps / periods


    my_feature = input_feature
    my_feature_data = california_housing_data_frame[[my_feature]]

    my_label = "median_house_value"
    targets = california_housing_data_frame[my_label]

    feature_columns = [tf.feature_column.numeric_column(my_feature)]

    training_input_fn = lambda: my_input_fn(my_feature_data, targets, batch_size=batch_size)
    prediction_input_fn = lambda: my_input_fn(my_feature_data, targets, num_epochs=1, shuffle=False)


    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=my_optimizer
    )

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.title("Learned Line by Period")
    plt.ylabel(my_label)
    plt.xlabel(my_feature)

    sample = california_housing_data_frame.sample(n=300)

    plt.scatter(sample[my_feature], sample[my_label])
    colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]

    print("Traning model...")
    print("RMSE (on training data):")
    root_mean_squared_errors = []
    for period in range(0, periods):
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_periods
        )

        predictions = linear_regressor.predict(input_fn=prediction_input_fn)
        predictions = np.array([item['predictions'][0] for item in predictions])

        root_mean_squared_error = math.sqrt(metrics.mean_squared_error(predictions, targets))

        print("period %02d: %02f" % (period, root_mean_squared_error))

        root_mean_squared_errors.append(root_mean_squared_error)

        y_extents = np.array([0, sample[my_label].max()])

        weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
        bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

        x_extents = (y_extents - bias) / weight
        x_extents = np.maximum(np.minimum(x_extents,
                                          sample[my_feature].max()),
                               sample[my_feature].min())
        y_extents = weight * x_extents + bias
        plt.plot(x_extents, y_extents, color=colors[period])


    print("Model training finished")
    plt.subplot(1, 2, 2)
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(root_mean_squared_errors)
    plt.show()

    # calibration_data = pd.DataFrame()
    # calibration_data["predictions"] = pd.Series(predictions)
    # calibration_data["targets"] = pd.Series(targets)
    # display.display(calibration_data.describe())
    #
    # print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)



train_model(
    learning_rate=0.0001,
    steps=150,
    batch_size=10,
    # input_feature='population'
)