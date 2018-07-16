

import math
import pandas as pd
from IPython import display
from matplotlib import  cm
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.data import Dataset
from sklearn import metrics


tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format


def preprocess_features(california_housing_data_frame):

    selected_features = california_housing_data_frame[
        ["latitude",
         "longitude",
         "housing_median_age",
         "total_rooms",
         "total_bedrooms",
         "population",
         "median_income"]]

    processed_features = selected_features.copy()

    processed_features["room_per_persion"] = (california_housing_data_frame["total_rooms"]/california_housing_data_frame["population"])

    return processed_features


def preprocess_targets(california_housing_data_frame):

    output_targets = pd.DataFrame()
    output_targets["median_house_value"] = (california_housing_data_frame["median_house_value"] / 1000.0)
    return output_targets


california_housing_data_frame = pd.read_csv("/Users/txg/Documents/california_housing_train.csv", sep=",")

california_housing_data_frame = california_housing_data_frame.reindex(np.random.permutation
                                                                      (california_housing_data_frame.index))

training_examples = preprocess_features(california_housing_data_frame=california_housing_data_frame.head(12000))

training_targets = preprocess_targets(california_housing_data_frame=california_housing_data_frame.head(12000))

validation_examples = preprocess_features(california_housing_data_frame.tail(5000))

validation_targets = preprocess_targets(california_housing_data_frame=california_housing_data_frame.tail(5000))


# plt.figure(figsize=(13, 8))
# ax = plt.subplot(1, 2, 1)
# ax.set_ylim([32, 43])
# ax.set_title("Validation Data")
# ax.set_autoscaley_on(False)
# ax.set_xlim([-126, -112])
# plt.scatter(validation_examples["longitude"], validation_examples["latitude"])
# ax = plt.subplot(1, 2, 2)
# ax.set_title("Training Data")
# ax.set_autoscaley_on(False)
# ax.set_ylim([32, 43])
# ax.set_autoscalex_on(False)
# ax.set_xlim([-126, -112])
# plt.scatter(training_examples["longitude"], training_examples["latitude"])
#
# plt.show()


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):

    features = {key:np.array(value) for (key, value) in dict(features).items()}

    ds = Dataset.from_tensor_slices((features, targets))

    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(10000)


    features, labels = ds.make_one_shot_iterator().get_next()
    return  features, labels


def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features])


def train_model(
    learning_rate,
    steps,
    batch_size,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):


    periods = 10
    steps_per_period = steps / periods

    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)


    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=construct_feature_columns(training_examples),
        optimizer=my_optimizer
    )

    training_input_fn = lambda : my_input_fn(training_examples, training_targets["median_house_value"])

    predict_training_input_fn = lambda : my_input_fn(training_examples, training_targets["median_house_value"], num_epochs=1, shuffle=False)

    predict_validation_input_fn = lambda : my_input_fn(validation_examples, validation_targets["median_house_value"], shuffle=False, num_epochs=1)


    print("Training model ...")
    print("RMSE (on training data):")

    training_rmse = []
    validation_rmse = []

    for period in range(0, periods):
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period,
        )

        training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item["predictions"][0] for item in training_predictions])

        validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item["predictions"][0] for item in validation_predictions])

        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(training_predictions, training_targets)
        )

        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets)
        )

        print("period %02d: %0.2f" % (period, training_root_mean_squared_error))

        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)

    print("Model training finished.")

    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()
    plt.show()

    return linear_regressor


linear_regressor = train_model(
    learning_rate=0.00003,
    steps=500,
    batch_size=5,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets
)

california_housing_test_data = pd.read_csv("/Users/txg/Documents/california_housing_test.csv", sep=",")

test_examples = preprocess_features(california_housing_test_data)
test_targets = preprocess_targets(california_housing_test_data)

test_input_fn = lambda :my_input_fn(test_examples, test_targets["median_house_value"],shuffle=False,num_epochs=1)


test_predictions = linear_regressor.predict(input_fn=test_input_fn)
test_predictions = np.array([item['predictions'][0] for item in test_predictions])

root_mean_squared_error = math.sqrt(
    metrics.mean_squared_error(test_predictions, test_targets)
)

print("Final RMSE (on test data):%0.2f" % root_mean_squared_error)


