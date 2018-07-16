
import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset


tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_data_frame = pd.read_csv("/Users/txg/Documents/california_housing_train.csv", sep=",")

california_housing_data_frame = california_housing_data_frame.reindex(np.random.permutation(california_housing_data_frame.index))

def preprocess_features(california_housing_data_frame):

    selected_features = california_housing_data_frame[
        ["latitude",
         "longitude",
         "housing_median_age",
         "total_rooms",
         "total_bedrooms",
         "population",
         "households",
         "median_income"]]

    processed_features = selected_features.copy()

    processed_features["rooms_per_person"] = (california_housing_data_frame["total_rooms"]/california_housing_data_frame["population"])

    return processed_features


def preprocess_targets(california_housing_data_frame):

    output_targets = pd.DataFrame()
    output_targets["median_house_value"] = (california_housing_data_frame["median_house_value"] / 1000.0)
    return output_targets

training_examples = preprocess_features(california_housing_data_frame=california_housing_data_frame.head(12000))

training_targets = preprocess_targets(california_housing_data_frame=california_housing_data_frame.head(12000))

validation_examples = preprocess_features(california_housing_data_frame.tail(5000))

validation_targets = preprocess_targets(california_housing_data_frame=california_housing_data_frame.tail(5000))


def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features])



def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):

    features = {key:np.array(value) for (key, value) in dict(features).items()}

    ds = Dataset.from_tensor_slices((features, targets))

    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(10000)


    features, labels = ds.make_one_shot_iterator().get_next()
    return  features, labels


def train_model(
    learning_rate,
    steps,
    batch_size,
    feature_columns,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):


    periods = 10
    steps_per_period = steps / periods

    my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)


    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
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

# _ = train_model(
#     learning_rate=1.0,
#     steps=500,
#     batch_size=100,
#     feature_columns=construct_feature_columns(training_examples),
#     training_examples=training_examples,
#     training_targets=training_targets,
#     validation_examples=validation_examples,
#     validation_targets=validation_targets)


def get_quantile_based_boundaries(feature_values, num_buckets):
    boundaries = np.arange(1.0, num_buckets) / num_buckets
    quantiles = feature_values.quantile(boundaries)
    return [quantiles[q] for q in quantiles.keys()]

# households = tf.feature_column.numeric_column("households")
# bucketized_households = tf.feature_column.bucketized_column(
#     households, boundaries=get_quantile_based_boundaries(
#         california_housing_data_frame["households"], 7
#     )
# )
#
#
# longitude = tf.feature_column.numeric_column("longitude")
# bucketized_longitude = tf.feature_column.bucketized_column(
#     longitude, boundaries=get_quantile_based_boundaries(
#         california_housing_data_frame["longitude"], 10
#     )
# )

def construct_feature_columns():

    households = tf.feature_column.numeric_column("households")
    longitude = tf.feature_column.numeric_column("longitude")
    latitude = tf.feature_column.numeric_column("latitude")
    housing_median_age = tf.feature_column.numeric_column("housing_median_age")
    median_income = tf.feature_column.numeric_column("median_income")
    rooms_per_person = tf.feature_column.numeric_column("rooms_per_person")

    bucketized_households = tf.feature_column.bucketized_column(
        households, boundaries=get_quantile_based_boundaries(
            training_examples["households"], 7
        )
    )

    bucketized_longitude = tf.feature_column.bucketized_column(
        longitude, boundaries=get_quantile_based_boundaries(
            training_examples["longitude"], 10
        )
    )

    bucketized_latitude = tf.feature_column.bucketized_column(
        latitude, boundaries=get_quantile_based_boundaries(
            training_examples["latitude"], 10
        )
    )

    bucketized_housing_median_age = tf.feature_column.bucketized_column(
        housing_median_age, boundaries=get_quantile_based_boundaries(
            training_examples["housing_median_age"], 7
        )
    )

    bucketized_median_incom = tf.feature_column.bucketized_column(
        median_income, boundaries=get_quantile_based_boundaries(
            training_examples["median_income"], 7
        )
    )

    bucketized_per_person = tf.feature_column.bucketized_column(
        rooms_per_person, boundaries=get_quantile_based_boundaries(
            training_examples["rooms_per_person"], 7
        )
    )

    feature_columns = set([
        bucketized_households,
        bucketized_housing_median_age,
        bucketized_longitude,
        bucketized_latitude,
        bucketized_median_incom,
        bucketized_per_person
    ])

    return feature_columns



_ = train_model(
    learning_rate=2,
    steps=5000,
    batch_size=100,
    feature_columns=construct_feature_columns(),
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)