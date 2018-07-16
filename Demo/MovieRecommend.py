import pandas as pd
import numpy as np
import tensorflow as tf

rating_df = pd.read_csv("../data/ml-latest-small/ratings.csv")


movies_df = pd.read_csv("../data/ml-latest-small/movies.csv")


movies_df["movieRow"] = movies_df.index

movies_df = movies_df[["movieRow", "movieId", "title"]]

# movies_df.to_csv("../data/ml-latest-small/moviesProcessed.csv", index=False, header=True, encoding='utf-8')

rating_df = pd.merge(rating_df, movies_df, on="movieId")


rating_df = rating_df[["userId", "movieRow", "rating"]]

# rating_df.to_csv("../data/ml-latest-small/ratingsProcessed.csv", index=False, header=True, encoding='utf-8')

userNo = rating_df['userId'].max() + 1
movieNo = rating_df['movieRow'].max() + 1

rating = np.zeros((movieNo, userNo))  ##创建一个movieNo * userNo 的空矩阵

flag = 0

rating_df_length = np.shape(rating_df)[0]  ##np.shape 查看矩阵的维数 x * y

for index, row in rating_df.iterrows():
    rating[int(row['movieRow']), int(row['userId'])] = row['rating']
    flag += 1


record = rating > 0

record = np.array(record, dtype=int)


def normalizeRatings(rating, record):
    m, n = rating.shape
    rating_mean = np.zeros((m, 1))
    rating_norm = np.zeros((m, n))

    for i in range(m):
        idx = record[i, :] != 0
        rating_mean[i] = np.mean(rating[i, idx])
        rating_norm[i, idx] -= rating_mean[i]

    return rating_norm, rating_mean


rating_norm, rating_mean = normalizeRatings(rating, record)

rating_norm = np.nan_to_num(rating_norm)

num_features = 10
X_parameters = tf.Variable(tf.random_normal([movieNo, num_features],stddev = 0.35))
Theta_parameters = tf.Variable(tf.random_normal([userNo, num_features],stddev = 0.35))

loss = 1/2 * tf.reduce_sum(((tf.matmul(X_parameters, Theta_parameters, transpose_b = True) - rating_norm) * record) ** 2) + 1/2 * (tf.reduce_sum(X_parameters ** 2) + tf.reduce_sum(Theta_parameters ** 2))

optimizer = tf.train.AdamOptimizer(1e-4)
train = optimizer.minimize(loss)

tf.summary.scalar('loss',loss)

