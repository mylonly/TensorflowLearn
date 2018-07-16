import pandas as pd
import tensorflow as tf


city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])

populations = pd.Series([854269, 1015785, 485199])

data = pd.DataFrame({'City Name': city_names, "Population": populations})

print(data)

print(data['City Name'])

print(data[['City Name']])




california_housing_data_frame = pd.read_csv("/Users/txg/Downloads/california_housing_train.csv", sep=",")


print(california_housing_data_frame.head(10))


my_feature = california_housing_data_frame.head(10)["total_rooms"]
print(my_feature)

my_feature = california_housing_data_frame.head(10)[["total_rooms"]]
print(my_feature)

