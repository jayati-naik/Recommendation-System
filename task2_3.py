import math
import sys
import json
import time
import numpy as np
import xgboost as xgb

from pyspark import SparkConf, SparkContext

def read_file(filepath):
    rdd = sc.textFile(filepath)
    return rdd


def calculate_avg(x, user_dict):
    num_of_ratings = len(x[1])
    sum = 0
    for i in x[1]:
        sum = sum + float(i[1])

    avg = sum / num_of_ratings
    return user_dict.get(x[0]), float(avg)


def calculate_numerator_denominator(corated_users, business_i, business_j, avg_rating_i, avg_rating_j):
    numerator = 0
    denominator_i = 0
    denominator_j = 0

    for user in corated_users:
        user_rating_i = business_i.get(user)
        user_rating_j = business_j.get(user)

        numerator += (user_rating_i - avg_rating_i) * (user_rating_j - avg_rating_j)

        denominator_i += ((user_rating_i - avg_rating_i) ** 2)
        denominator_j += ((user_rating_j - avg_rating_j) ** 2)

    denominator = math.sqrt(denominator_i) * math.sqrt(denominator_j)

    return numerator,denominator


def calculate_similarity_quoteint(similarity, avg_rating_i, avg_rating_j, cur_user_rating):
    avg_diff = abs(avg_rating_i - avg_rating_j)
    if 0 <= avg_diff <= 1:
        similarity_q = 1.0
        similarity.append([similarity_q, similarity_q*cur_user_rating, abs(similarity_q)])
    elif 1 < avg_diff <= 2:
        similarity_q = 0.5
        similarity.append([similarity_q, similarity_q*cur_user_rating, abs(similarity_q)])
    else:
        similarity_q = 0.0
        similarity.append([similarity_q, similarity_q*cur_user_rating, abs(similarity_q)])

    return similarity


def calculate_similarity(cur_user, cur_business, business_rated_cur_user, u_b_dict, business_avg_ratings):
    similarity = []
    business_i = dict(u_b_dict.get(cur_business))
    users_rated_c = set(business_i.keys())
    ci_avg = business_avg_ratings[cur_business]
    avg_rating_i = ci_avg[1]

    for y, business in enumerate(business_rated_cur_user):
        business_j = dict(u_b_dict.get(business))
        users_rated_b = set(business_j.keys())
        bj_avg = business_avg_ratings[business]
        avg_rating_j = bj_avg[1]
        cur_user_rating = business_j.get(cur_user)

        corated_users = users_rated_b & users_rated_c

        if len(corated_users) > 1:
            numerator, denominator = calculate_numerator_denominator(corated_users, business_i, business_j, avg_rating_i, avg_rating_j)
            if numerator == 0 or denominator == 0:
                similarity_q = 0.0
                similarity.append([similarity_q, similarity_q*cur_user_rating, abs(similarity_q)])
            elif numerator < 0 or denominator < 0:
                continue
            else:
                similarity_q = numerator/denominator
                similarity.append([similarity_q, similarity_q*cur_user_rating, abs(similarity_q)])
        else:
            similarity = calculate_similarity_quoteint(similarity, avg_rating_i, avg_rating_j, cur_user_rating)

    return similarity


def calculate_predictions(x, train_users, user_list, train_businesses, business_list, utility_user_dict, utility_business_dict, business_avg_ratings, neighbours):
    # Calculate similarity between pair of business_ids
    cur_user, cur_business = x[0], x[1]
    if cur_business not in business_list:
        return 3.0

    if cur_user not in user_list:
        return 3.0

    rated_cur_user_dict = dict(utility_user_dict.get(train_users.get(cur_user)))
    business_rated_cur_user = rated_cur_user_dict.keys()

    # Calculate Similarity with other businesses
    similar_businesses = calculate_similarity(train_users.get(cur_user), train_businesses.get(cur_business), business_rated_cur_user, utility_business_dict, business_avg_ratings)

    # Sort the list of similarity by value and pick top business_ids
    similar_businesses.sort(reverse=True)
    similar_businesses = similar_businesses[:neighbours]

    similar_businesses_np = np.array(similar_businesses)
    sum_similar_businesses = similar_businesses_np.sum(axis=0)

    if sum_similar_businesses[1] == 0.0 or sum_similar_businesses[2] == 0.0:
        return 0.0

    prediction = sum_similar_businesses[1]/sum_similar_businesses[2]

    return prediction


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
    .map(lambda x: (str(x[0]), str(x[1]), float(x[2])))

header = yelp_test_user_business.first()
yelp_test_user_business = yelp_test_user_business.filter(lambda row: row != header)
yelp_test_user_business = yelp_test_user_business.map(lambda line: line.split(",")) \
    .map(lambda x: (str(x[0]), str(x[1]), float(x[2])))

# Task 2_1
# Encode users and business ids.
user_ids = yelp_train_user_business.map(lambda x: (x[0], 1)).reduceByKey(lambda x, y: x).map(lambda x: x[0]).distinct()
users = list(user_ids.collect())
users.sort()
user_dict = {}
for i, u in enumerate(list(users)):
    user_dict[u] = i

business_ids = yelp_train_user_business.map(lambda x: (x[1], 1)).reduceByKey(lambda x, y: x).map(
    lambda x: x[0]).distinct()
businesses = list(business_ids.collect())
businesses.sort()
business_dict = {}
for i, b in enumerate(list(businesses)):
    business_dict[b] = i

# Make Business Utility Matrix
utility_business_matrix = yelp_train_user_business.map(lambda x: (business_dict.get(x[1]), [(user_dict.get(str(x[0])), x[2])])) \
    .reduceByKey(lambda x, y: x + y) \
    .map(lambda x: x) \
    .sortBy(lambda x: x[0])

utility_business_dict = {}
for i, j in utility_business_matrix.collect():
    utility_business_dict[i] = j

# Make User Utility Matrix
utility_user_matrix = yelp_train_user_business.map(lambda x: (user_dict.get(x[0]), [(business_dict.get(str(x[1])), x[2])])) \
    .reduceByKey(lambda x, y: x + y) \
    .map(lambda x: x) \
    .sortBy(lambda x: x[0])

utility_user_dict = {}
for i, j in utility_user_matrix.collect():
    utility_user_dict[i] = j

# Calculate average rating for business
# Calculate co-rated weight
business_avg_ratings = utility_business_matrix.map(lambda x: calculate_avg(x, user_dict)).collect()

neighbours=80
# Calculate weighted average for pair of business ids
predictions_Item_based = yelp_test_user_business.map(lambda x: calculate_predictions(x, user_dict, users, business_dict, businesses,
                                                                                     utility_user_dict, utility_business_dict, business_avg_ratings, neighbours)).collect()

predictions_Item_based_np = np.asarray(predictions_Item_based, dtype = 'float')

# Task 2_2
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
predictions_xgboost = xgbModel.predict(test_data_x)

predictions_xgboost_to_be_printed = np.c_[test_data_np[:, : 2], predictions_xgboost]

# Make the weighted predictions.
hybrid_prediction = 0.10 * predictions_Item_based_np + 0.90 * predictions_xgboost

predictions_to_be_printed = np.c_[test_data_np[:, : 2], hybrid_prediction]

# Write contents into file
file = open(output_file_name, 'w')
file.write("user_id, business_id, prediction")
file.write("\n")
for p in predictions_to_be_printed:
    file.write(str(p[0]) + "," + str(p[1]) + "," + str(p[2]) + "\n")
file.close()

stop_time = time.time()
print("Duration" + str(stop_time - start_time))