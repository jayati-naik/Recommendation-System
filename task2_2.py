import json
import sys
import time
import xgboost as xgb

import numpy as np
from pyspark import SparkConf, SparkContext


def read_file(filepath):
    rdd = sc.textFile(filepath)
    return rdd


def selectFeatures(x, user_data, business_data, test_data):
    user, business = x[0], x[1]
    rating = 0

    if test_data:
        rating = x[2]
    else:
        rating = -1.0

    if (user not in user_data.keys() or business not in business_data.keys()):
        return [user, business, None, None, None, None, None]

    user_review_count, user_average_stars = user_data.get(user)
    business_review_count, business_average_stars = business_data.get(business)

    return[str(user), str(business), float(rating), float(user_review_count), float(user_average_stars), float(business_review_count), float(business_average_stars)]


conf = SparkConf().setAppName("HW3-Task2_1")
sc = SparkContext(conf=conf)
sc.setLogLevel('ERROR')

folder_path = sys.argv[1]
train_file_name = sys.argv[2]
output_file_name = sys.argv[3]

start_time = time.time()
# Read file into an RDD
yelp_train_user_business = read_file(train_file_name)
yelp_test_user_business = read_file(folder_path + 'yelp_train.csv')

# Remove header
# Read User_id business_id ratings
header = yelp_train_user_business.first()
yelp_train_user_business = yelp_train_user_business.filter(lambda row: row != header)
yelp_train_user_business = yelp_train_user_business.map(lambda line: line.split(","))

header = yelp_test_user_business.first()
yelp_test_user_business = yelp_test_user_business.filter(lambda row: row != header)
yelp_test_user_business = yelp_test_user_business.map(lambda line: line.split(","))

# Read feature files
user = read_file(folder_path + 'user.json')
user_data = user.map(lambda x: json.loads(x)).map(lambda x: (
    (str(x['user_id'])), (float(x['review_count']), float(x['average_stars'])))).collectAsMap()

business = read_file(folder_path + 'business.json')
business_data = business.map(lambda x: json.loads(x)).map(
    lambda x: ((str(x['business_id'])), (float(x['stars']), float(x['review_count'])))).collectAsMap()

training_data = yelp_train_user_business.map(lambda x: selectFeatures(x, user_data, business_data, False)).collect()
training_data_np = np.array(training_data)

training_data_x, training_data_y = training_data_np[:, 2: -1], training_data_np[:, -1]
train_data_x, training_data_y = np.array(training_data_x, dtype='float'), np.array(training_data_y, dtype='float')

xgbModel = xgb.XGBRegressor(objective='reg:linear')
xgbModel.fit(training_data_x, training_data_y)

test_data = yelp_test_user_business.map(lambda x: selectFeatures(x, user_data, business_data, True)).collect()
test_data_np = np.array(test_data)

test_data_x, test_data_y = test_data_np[:, 2: -1], test_data_np[:, -1]
test_data_x, test_data_y = np.array(test_data_x, dtype='float'), np.array(test_data_y, dtype='float')

modelPreds = xgbModel.predict(test_data_x)

predictVals = np.c_[test_data_x[:, : 2], modelPreds]

# Write contents into file
file = open(output_file_name, 'w')
file.write("user_id, business_id, prediction")
file.write("\n")
for p in predictVals:
    file.write(str(p[0]) + "," + str(p[1]) + "," + str(p[2]) + "\n")
file.close()

stop_time = time.time()
print("Duration" + str(stop_time - start_time))

