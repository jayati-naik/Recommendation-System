import csv
import sys
import time

from pyspark import SparkConf, SparkContext


def read_file(filepath):
    rdd = sc.textFile(input_file_path)
    return rdd


def generate_hash_functions(num_hash, num_users):
    def h(x):
        if num_hash % 2 == 0:
            return min((num_hash * c + 1) % num_users for c in x[1])
        else:
            return min((num_hash * c) % num_users for c in x[1])
    return h


def divide_sm_into_bands(x):
    band = []
    for i in range(b):
        band.append(((i, tuple(x[1][int(i * r):int(i * r + r)])), [x[0]]))
    return band


def generate_pairs(x):
    list_business_ids = list(x[1])
    list_business_ids.sort()
    candidate_pairs = []

    for i in range(len(list_business_ids)):
        for j in range(i + 1, len(list_business_ids)):
            candidate_pairs.append(((list_business_ids[i], list_business_ids[j]), 1))

    return candidate_pairs


def calculate_jaccard_similarity(x):
    b1 = set(cm_list[business_id_dict[x[0]]][1])
    b2 = set(cm_list[business_id_dict[x[1]]][1])
    i = list(b1 & b2)
    u = list(b1 | b2)
    j_similarity = float(len(i))/float(len(u))

    return (x[0], x[1], j_similarity)


conf = SparkConf().setAppName("HW3-Task1")
sc = SparkContext(conf=conf)
sc.setLogLevel('ERROR')

cmd_args = str(sys.argv)
cmd_args = cmd_args.split(", ")
input_file_path = cmd_args[1].replace("'", "")
output_file_path = cmd_args[2].replace("'", "").replace(']', '')

start_time = time.time()

# Read file into an RDD
yelp_user_business = read_file(input_file_path)
header = yelp_user_business.first()
yelp_user_business = yelp_user_business.filter(lambda row: row != header)

# Fetch user ids and business ids
yelp_user_business = yelp_user_business.map(lambda line: line.split(",")) \
    .map(lambda x: (str(x[0]), str(x[1]), str(x[2])))

print("Generate Characteristic Matrix: " + str(time.time() - start_time))
# Generate Characteristic Matrix
user_ids = yelp_user_business.map(lambda x: (x[0], 1)).reduceByKey(lambda x, y: x).map(lambda x: x[0]).distinct()
users = list(user_ids.collect())
users.sort()
user_id_dict = {}
for i, u in enumerate(list(users)):
    user_id_dict[u] = i

business_ids = yelp_user_business.map(lambda x: (x[1], 1)).reduceByKey(lambda x, y: x).map(lambda x: x[0]).distinct()
businesses = list(business_ids.collect())
businesses.sort()
business_id_dict = {}
for i, b in enumerate(list(businesses)):
    business_id_dict[b] = i

characteristic_matrix = yelp_user_business.map(lambda x: (str(x[1]), [user_id_dict.get(str(x[0]))])) \
    .reduceByKey(lambda x, y: x + y) \
    .map(lambda x: x) \
    .sortBy(lambda x: x[0])

cm_list = characteristic_matrix.collect()

# Generate Hash function Columns to Signature matrix
num_hashes = 150
hash_functions = []
for i in range(num_hashes):
    hash_functions.append(generate_hash_functions(i, len(users)))

signature_matrix = characteristic_matrix.map(lambda x: (x[0], [h(x) for h in hash_functions]))

# Divide signature matrix into bands and rows st. b X r = num_hashes
# Calculate candidate pairs
b = 75
r = (num_hashes/b)
candidates = signature_matrix.flatMap(divide_sm_into_bands).reduceByKey(lambda x, y: x + y)\
    .filter(lambda x: len(x[1]) >= 2)

# Generate pairs
candidates = candidates.flatMap(generate_pairs).reduceByKey(lambda x, y: x).map(lambda x: x[0])

# Calculate Jaccard Similarity
similar_itemsets = candidates.map(calculate_jaccard_similarity).filter(lambda x: x[2] >= 0.5).map(lambda x: (x[0], x[1], str(x[2]))).sortBy(lambda x: (x[0], x[1], x[2]))

# Write contents into file
file = open(output_file_path, 'w')
file.write("business_id_1, business_id_2, similarity")
file.write("\n")
for s in similar_itemsets.collect():
    file.write(str(s[0]+","+s[1]+","+s[2])+"\n")
file.close()

stop_time = time.time()
print("Duration" + str(stop_time - start_time))

business_pairs = similar_itemsets.map(lambda x : (x[0], x[1]))
comparision_file = "/Users/jayati/Projects/hw3/data/pure_jaccard_similarity.csv"
truthRdd = sc.textFile(comparision_file, minPartitions=None, use_unicode=False)
truthRdd = truthRdd.mapPartitions(lambda x : csv.reader(x))
truthRdd = truthRdd.map(lambda x : (str(x[0]), str(x[1])))

tp = float(business_pairs.intersection(truthRdd).count())
fp = float(business_pairs.subtract(truthRdd).count())
fn = float(truthRdd.subtract(business_pairs).count())
precision = tp/(tp+fp)
recall = tp/(tp+fn)

print("Precision", precision)
print("Recall", recall)