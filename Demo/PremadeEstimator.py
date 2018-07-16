import pandas as pd
import numpy as np
import tensorflow as tf

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']

train_df = pd.read_csv("../data/iris_training.csv", names=CSV_COLUMN_NAMES, header=0)
train_features = train_df[['SepalLength', 'SepalWidth',
                           'PetalLength', 'PetalWidth']]
train_targets = train_df['Species']

test_df = pd.read_csv("../data/iris_test.csv", names=CSV_COLUMN_NAMES, header=0)
test_features = test_df[['SepalLength', 'SepalWidth',
                        'PetalLength', 'PetalWidth']]
test_targets = test_df[['Species']]


##构建tensorflow 特征列

feature_columns = set([tf.feature_column.numeric_column(feature) for feature in train_features])

classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units= [10, 10],
    n_classes=3,
    model_dir='models/iris'
)

def train_input_fn(features, target, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), target))
    dataset = dataset.shuffle(buffer_size=1000).repeat(count=None).batch(batch_size=batch_size)
    return  dataset.make_one_shot_iterator().get_next()

def eval_input_fn(features, target=None, batch_size=None):
    if target is None:
        inputs = features
    else:
        inputs = (dict(features), target)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(batch_size=batch_size)
    return dataset.make_one_shot_iterator().get_next()

classifier.train(
    input_fn=lambda : train_input_fn(train_features, train_targets, 100),
    steps=100
)

eval_result = classifier.evaluate(
    input_fn=lambda : eval_input_fn(test_features, test_targets, 100)
)

print("Test set accuracy: %0.3f" % eval_result['accuracy'])

predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
}

predictions = classifier.predict(
    input_fn=lambda :eval_input_fn(predict_x, batch_size=100)
)


expected = ['Setosa', 'Versicolor', 'Virginica']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']


for pred_dict, expec in zip(predictions, expected):
    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    print(template.format(SPECIES[class_id], 100 * probability, expec))