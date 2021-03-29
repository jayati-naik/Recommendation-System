import sys
import json
import time
import numpy as np
import xgboost as xgb

from pyspark import SparkConf, SparkContext

def read_file(filepath):
    rdd = sc.textFile(filepath)
    return rdd


def generate_data(yelp_user_business, user_data_dict, business_data_dict, test_data=False):
    data = yelp_user_business.map(lambda x: select_features(x, user_data_dict, business_data_dict, test_data)).collect()
    data_np = np.array(data)
    data_x = data_np[:, 2: -1]
    data_y = data_np[:, -1]
    x = np.array(data_x, dtype='float')
    y = np.array(data_y, dtype='float')

    return x, y


def select_features(x, business_data_dict, user_data_dict, test_features=False):
    user, business = x[0], x[1]

    if (user not in user_data_dict.keys() or business not in business_data_dict.keys()):
        return [user, business, None, None, None, None, None]

    if test_features:
        rating = -1.0
    else:
        rating = x[2]

    user_review_count, user_average_stars = user_data_dict.get(user)
    business_review_count, business_average_stars = business_data_dict.get(business)

    return[str(user), str(business), float(business_review_count), float(business_average_stars), float(user_review_count), float(user_average_stars), float(rating)]

REGRESSION_LINEAR = 'reg:linear'

conf = SparkConf().setAppName("HW3-Task2_2")
sc = SparkContext(conf=conf)
sc.setLogLevel('ERROR')

folder_path = sys.argv[1]
test_file_name = sys.argv[2]
output_file_name = sys.argv[3]

start_time = time.time()
# Read file into an RDD
yelp_train_user_business = read_file(folder_path + 'yelp_train.csv')
yelp_test_user_business = read_file(test_file_name)

# Remove header
# Read User_id business_id ratings
header = yelp_train_user_business.first()
yelp_train_user_business = yelp_train_user_business.filter(lambda row: row != header)
yelp_train_user_business = yelp_train_user_business.map(lambda line: line.split(",")) \
    .map(lambda x: ((str(x[0]), str(x[1]), float(x[2]))))

header = yelp_test_user_business.first()
yelp_test_user_business = yelp_test_user_business.filter(lambda row: row != header)
yelp_test_user_business = yelp_test_user_business.map(lambda line: line.split(",")) \
    .map(lambda x: (str(x[0]), str(x[1])))

# Read feature files
business = read_file(folder_path + 'business.json')
business_data = business.map(lambda x: json.loads(x)).map(lambda x: (
    (str(x['business_id'])), (float(x['review_count']), float(x['stars'])))).collect()
business_data_dict = dict(business_data)

user = read_file(folder_path + 'user.json')
user_data = user.map(lambda x: json.loads(x)).map(lambda x: (
    (str(x['user_id'])), (float(x['review_count']), float(x['average_stars'])))).collect()
user_data_dict = dict(user_data)

# Generate Training Data
train_data_x, train_data_y = generate_data(yelp_train_user_business, user_data_dict, business_data_dict)

# Train model using training data
xgbModel = xgb.XGBRegressor(objective=REGRESSION_LINEAR)
xgbModel.fit(train_data_x, train_data_y)

# Generate Test Data
test_data = yelp_test_user_business.map(lambda x: select_features(x, user_data_dict, business_data_dict, True)).collect()
test_data_np = np.array(test_data)
test_data_x, test_data_y = generate_data(yelp_test_user_business, user_data_dict, business_data_dict, True)

# Predict model using training data
predictions = xgbModel.predict(test_data_x)

predictions_to_be_printed = np.c_[test_data_np[:, : 2], predictions]

# Write contents into file
file = open(output_file_name, 'w')
file.write("user_id, business_id, prediction")
file.write("\n")
for p in predictions_to_be_printed:
    file.write(str(p[0]) + "," + str(p[1]) + "," + str(p[2]) + "\n")
file.close()

stop_time = time.time()
print("Duration" + str(stop_time - start_time))