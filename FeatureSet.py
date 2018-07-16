
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
         "median_income"]]

    processed_features = selected_features.copy()

    processed_features["room_per_persion"] = (california_housing_data_frame["total_rooms"]/california_housing_data_frame["population"])

    return processed_features


def preprocess_targets(california_housing_data_frame):

    output_targets = pd.DataFrame()
    output_targets["median_house_value"] = (california_housing_data_frame["median_house_value"] / 1000.0)
    return output_targets


training_examples = preprocess_features(california_housing_data_frame=california_housing_data_frame.head(12000))

training_targets = preprocess_targets(california_housing_data_frame=california_housing_data_frame.head(12000))

validation_examples = preprocess_features(california_housing_data_frame.tail(5000))

validation_targets = preprocess_targets(california_housing_data_frame=california_housing_data_frame.tail(5000))


correlation_data_frame = training_examples.copy()
correlation_data_frame["target"] = training_targets["median_house_value"]

correlation_data_frame.corr()


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


# LATITUDE_RANGES = zip(range(32, 44), range(33, 45))
#
#
# def select_and_transform_features(source_df):
#     selected_examples = pd.DataFrame()
#     selected_examples["median_income"] = source_df["median_income"]
#     for r in LATITUDE_RANGES:
#         selected_examples["latitude_%d_to_%d" % r] = source_df['latitude'].apply(lambda l: 1.0 if 1 >= r[0] and 1 < r[1] else 0.0)
#     return  selected_examples
#
#
# selected_training_examples = select_and_transform_features(training_examples)
# selected_validation_examples = select_and_transform_features(validation_examples)
#
LATITUDE_RANGES = zip(range(32, 44), range(33, 45))

LATITUDE_RANGES = list(LATITUDE_RANGES)


def select_and_transform_features(source_df):
  selected_examples = pd.DataFrame()
  selected_examples["median_income"] = source_df["median_income"]
  selected_examples["room_per_persion"] = (
          source_df["total_rooms"] / source_df["population"])

  for r in LATITUDE_RANGES:
    selected_examples["latitude_%d_to_%d" % r ] = source_df["latitude"].apply(
      lambda l: 1.0 if l >= r[0] and l < r[1] else 0.0)

  return selected_examples

###zip 函数有问题，循环遍历
###在python3 下zip返回的是迭代器

selected_training_examples = select_and_transform_features(training_examples)
selected_validation_examples = select_and_transform_features(validation_examples)


# minimal_features = ["median_income", "latitude"]
#
# assert  minimal_features, "You must select at least one feature"
#
# selected_training_examples = training_examples[minimal_features]
# selected_validation_examples = validation_examples[minimal_features]

train_model(
    learning_rate=0.01,
    steps=5000,
    batch_size=10,
    training_examples=selected_training_examples,
    training_targets=training_targets,
    validation_examples=selected_validation_examples,
    validation_targets=validation_targets
)

