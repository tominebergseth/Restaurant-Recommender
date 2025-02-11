# Tomine Bergseth, DSCI 553 competition final project

################################################################
# Method Description:
# ----------------------
# After experimenting with a handful of approaches for this assignment, I ended up going with an
# optimized model-based recommendation system using XGBoost’s XGBRegressor due to its combined
# efficiency and performance. Initially I wanted to cluster the data and add cluster labels as a feature for the
# regression model to see if it would enhance performance; however, this was not efficient enough. That being said,
# in the process I started working on feature engineering and discovered it significantly improved the performance
# of my model-based recommender from homework 3. For instance, I originally filtered out closed businesses as
# I figured we would not be asked to recommend closed businesses, but by removing this filter I reduced my RMSE.

# For feature engineering, I went through all the available data files and created features from everything I could think of
# and removed it if it hurt performance or kept it if it improved it at all.  This led to a lot more features than
# I originally had including average business rating, business review count, whether the business is open, count
# of business attributes, count of business categories, count of photos for a business, count of tips for a business
# and by a user, check-in counts by business, average stars and review count by user, sum of funny, cool,
# useful for user, sum of compliments for user, and fan and friend count for user. 
# Finally, I used RandomSearchCV and then GridSearchCV localy to fine-tune parameters for the model which helped optimize
# my model and get it the final points below .98 for RMSE.

# ----------------------------------------------------------------------------------------------------------------------
# Error Distribution:
# -------------------
# >=0 and <1   n = 102326
# >=1 and <2   n = 32796
# >=2 and <3   n = 6121
# >=3 and <4   n = 801
# >=4          n = 0

# ----------------------------------------------------------------------------------------------------------------------
# RMSE:
# ---------
## Validation data: 0.9778021732643613 

# -----------------------------------------------------------------------------------------------------------------------
# Execution time:
## 594.4517555236816 seconds (on Vocareum)
# ----------------------------------------------------------------------------------------------------------------------


# imports
from pyspark import SparkContext
import sys
import os
import time
import json
import numpy as np
from xgboost import XGBRegressor
import statistics
from collections import Counter
# from sklearn.preprocessing import MinMaxScaler


# setting up os environment variables as done for hw0
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

sc = SparkContext('local[*]', 'comp')  # initialize spark context
sc.setLogLevel("WARN")  # setting loglevel


def compute_stats(ratings):
    # convert ratings to a list
    ratings_list = list(ratings)
    # print(ratings_list)
    
    # calculate mean
    mean = sum(ratings_list) / len(ratings_list)
    
    # calculate variance
    variance = sum((x - mean) ** 2 for x in ratings_list) / len(ratings_list)
    
    # calculate standard deviation
    std_dev = variance ** 0.5
    
    median = statistics.median(ratings_list)
    
    ratings_list = list(ratings)
    count = Counter(ratings)
    max_count = max(count.values())
    mode = next(k for k, v in count.items() if v == max_count)

    return [std_dev, median, mode, len(ratings_list)]


def add_features(folder_path, file, train):
    # load yelp_train file
    if file == "yelp_train.csv":
        rdd = sc.textFile(os.path.join(folder_path, "yelp_train.csv"))
    else:
        rdd = sc.textFile(file)
    # rdd.cache()
    header = rdd.first()  # extracting the header
    rdd = rdd.filter(lambda x: x != header)  # filtering out the header row
    rdd = rdd.map(lambda x: x.split(","))  # split by comma

    # create features from business file
    rdd_business = sc.textFile(os.path.join(folder_path, "business.json"))
    # print(rdd_business.take(1))
    rdd_json_business = rdd_business.map(lambda x: json.loads(x))  # parse the json and return reach line as rdd
    # print(rdd_json_business.take(1))
    business_feat = rdd_json_business.map(lambda x: (x["business_id"],
                                                     [x["stars"], x["review_count"], x["is_open"],
                                                      len(x["attributes"]) if x["attributes"] else 0,
                                                      len(x["categories"].split(",")) if x["categories"] else 0]))
    # compute global business avg rating
    bavg_rdd = business_feat.map(lambda x: x[1][0])
    sum_count_b = bavg_rdd.map(lambda x: (x, 1)) \
                      .reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))
    b_mean = sum_count_b[0] / sum_count_b[1]

    # from photo file, add number of photos for each business
    rdd_photo = sc.textFile(os.path.join(folder_path, "photo.json"))
    rdd_json_photo = rdd_photo.map(lambda x: json.loads(x))  # parse the json and return reach line as rdd
    photo_feat = rdd_json_photo.map(lambda x: (x["business_id"], 1)).reduceByKey(lambda x, y: x + y).distinct()
    # print("photo", photo_feat.take(1))
    photo_map = photo_feat.collectAsMap()

    # from tip file create tip count feature by busioness and by user
    tip_rdd = sc.textFile(os.path.join(folder_path, "tip.json"))
    tip_rdd = tip_rdd.map(lambda x: json.loads(x))
    # print(tip_rdd.take(1))
    rdd_tip = tip_rdd.map(lambda x: (x["business_id"], x["user_id"]))
    # print(rdd_tip.take(1))
  
    rdd_tip_u = rdd_tip.map(lambda x: (x[1], x[0])).groupByKey().mapValues(lambda x: (len(list(x))))
    # , sum(likes for _, likes in x)))
    rdd_tip_b = rdd_tip.groupByKey().mapValues(lambda x: (len(list(x))))
    # print(rdd_tip_b.take(1))
    tip_b = rdd_tip_b.collectAsMap()
    tip_u = rdd_tip_u.collectAsMap()
    
    # read in and create features from user file
    rdd_user = sc.textFile(os.path.join(folder_path, "user.json"))
    rdd_json_user = rdd_user.map(lambda x: json.loads(x))  # parse the json and return reach line as rdd
    user_feat = rdd_json_user.map(lambda x: (x["user_id"],
                                             [x["review_count"], x["average_stars"],
                                              x["fans"], x["useful"] + x["funny"] + x["cool"],
                                              sum(v for k, v in x.items() if k.startswith("compliment")),
                                              # datetime.now().year - int(x["yelping_since"][:4])
                                              # if x["yelping_since"] else 0,
                                              len(x["friends"].split(", ")) if x["friends"] else 0,
                                              tip_u[x["user_id"]] if x["user_id"] in tip_u else 0]))
    # compute global user avg rating
    uavg_rdd = user_feat.map(lambda x: x[1][1])
    sum_count_u = uavg_rdd.map(lambda x: (x, 1)) \
                      .reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))
    user_mean = sum_count_u[0] / sum_count_u[1]

    # try to aggregate checkins by business id and add as feature
    checkin_rdd = sc.textFile(os.path.join(folder_path, "checkin.json"))
    rdd_checkin = checkin_rdd.map(lambda x: json.loads(x))
    checkin_feat = rdd_checkin.map(lambda x: (x["business_id"], sum(x["time"].values()))).reduceByKey(lambda a, b: a + b)
    # print(checkin_feat.take(1))
    checkin_map = checkin_feat.collectAsMap()

    # add in photo, checkin and tip features to business features
    business_f = business_feat.map(lambda x: (x[0], [x[1][0], x[1][1], x[1][2], x[1][3], x[1][4],
                                                      photo_map[x[0]] if x[0] in photo_map else 0,
                                                      checkin_map[x[0]] if x[0] in checkin_map else 0,
                                                      tip_b[x[0]] if x[0] in tip_b else 0]))
    # print(business_f.take(1))

    # now join in features to train and val/test data
    if file == "yelp_train.csv":
        joined_rdd = rdd.map(lambda x: (x[1], (x[0], float(x[2])))).leftOuterJoin(business_f).\
            map(lambda x: (x[1][0][0], (x[0], x[1][0][1], x[1][1])))
        # print(joined_rdd.take(1))
        joined_rdd = joined_rdd.leftOuterJoin(user_feat)
        # print(joined_rdd.take(1))
        joined_rdd = joined_rdd.map(lambda x: (x[0], x[1][0][0], x[1][0][1],
                                               [x[1][0][2][0] if x[1][0][2] is not None else b_mean,
                                                x[1][0][2][1] if x[1][0][2] is not None else 0,
                                                x[1][0][2][2] if x[1][0][2] is not None else np.nan,
                                                x[1][0][2][3] if x[1][0][2] is not None else np.nan,
                                                x[1][0][2][4] if x[1][0][2] is not None else np.nan,
                                                x[1][0][2][5] if x[1][0][2] is not None else np.nan,
                                                x[1][0][2][6] if x[1][0][2] is not None else np.nan,
                                                x[1][0][2][7] if x[1][0][2] is not None else np.nan,
                                                x[1][1][0] if x[1][1] is not None else 0,
                                                x[1][1][1] if x[1][1] is not None else user_mean,
                                                x[1][1][2] if x[1][1] is not None else np.nan,
                                                x[1][1][3] if x[1][1] is not None else np.nan,
                                                x[1][1][4] if x[1][1] is not None else np.nan,
                                                x[1][1][5] if x[1][1] is not None else np.nan,
                                                x[1][1][6] if x[1][1] is not None else np.nan]))
        # print(joined_rdd.take(1))
                                                                                           
    else:  # for submitting if test data does not have the target map test different
        joined_rdd = rdd.map(lambda x: (x[1], x[0])).leftOuterJoin(business_f).map(lambda x: (x[1][0], (x[0], x[1][1])))
        # print(joined_rdd.take(1))
        joined_rdd = joined_rdd.leftOuterJoin(user_feat)
        # print(joined_rdd.take(1))
            
        joined_rdd = joined_rdd.map(lambda x: (x[0], x[1][0][0],
                                               [x[1][0][1][0] if x[1][0][1] is not None else b_mean,
                                                x[1][0][1][1] if x[1][0][1] is not None else 0,
                                                x[1][0][1][2] if x[1][0][1] is not None else np.nan,
                                                x[1][0][1][3] if x[1][0][1] is not None else np.nan,
                                                x[1][0][1][4] if x[1][0][1] is not None else np.nan,
                                                x[1][0][1][5] if x[1][0][1] is not None else np.nan,
                                                x[1][0][1][6] if x[1][0][1] is not None else np.nan,
                                                x[1][0][1][7] if x[1][0][1] is not None else np.nan,
                                                x[1][1][0] if x[1][1] is not None else 0,
                                                x[1][1][1] if x[1][1] is not None else user_mean,
                                                x[1][1][2] if x[1][1] is not None else np.nan,
                                                x[1][1][3] if x[1][1] is not None else np.nan,
                                                x[1][1][4] if x[1][1] is not None else np.nan,
                                                x[1][1][5] if x[1][1] is not None else np.nan,
                                                x[1][1][6] if x[1][1] is not None else np.nan]))
        # print("join", joined_rdd.take(1))
    
    return joined_rdd


def encode_ids(rdd_train, rdd_test):
    """Function to create a numerical encoding of the user and business ids so they can be included in features"""
    # create encodings based on training data
    user_ids = rdd_train.map(lambda x: x[0]).union(rdd_test.map(lambda x: x[0])).distinct().zipWithIndex().collectAsMap()
    business_ids = rdd_train.map(lambda x: x[1]).union(rdd_test.map(lambda x: x[1])).distinct().zipWithIndex().collectAsMap()

    # encode train
    encoded_train = rdd_train.map(lambda x: (user_ids[x[0]], business_ids[x[1]], x[2], x[3]))
    # print(encoded_train.take(1))

    # encode test
    encoded_test = rdd_test.map(lambda x: (user_ids[x[0]], business_ids[x[1]], x[2], x[3]))
    # ****take out x[3
    # print(encoded_test.take(1))

    return encoded_train, encoded_test


def prep_features_target(encoded_train, encoded_test):
    features = encoded_train.map(lambda x: ([x[3][0], x[3][1], x[3][2], x[3][3], x[3][4], x[3][5], x[3][6],
                                             x[3][7], x[3][8], x[3][9], x[3][10], x[3][11], x[3][12],
                                             x[3][13], x[3][14]]))
    target = encoded_train.map(lambda x: x[2])
    # print(features.take(1))
    # print(target.take(1))

    # convert features and target rdds to numpy array
    X_train = np.array(features.collect())
    y_train = np.array(target.collect())

    test_feat = encoded_test.map(lambda x:  ([x[2][0], x[2][1], x[2][2], x[2][3], x[2][4], x[2][5], x[2][6], x[2][7],
                                              x[2][8], x[2][9], x[2][10], x[2][11], x[2][12], x[2][13], x[2][14]]))
    # test_target = encoded_test.map(lambda x: x[2])
    X_test = np.array(test_feat.collect())
    # y_test = np.array(test_target.collect())
    
    # scaler = MinMaxScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    return X_train, y_train, X_test  # , y_test


def model_predictions(X_train, y_train, X_test):
    # initialize params dict with values i discover from local gridsearch/randomsearchcv
    params_dict = {'subsample': 0.8,
                   'reg_lambda': 10,
                   'reg_alpha': 0.5,
                   'n_estimators': 400,
                   'min_child_weight': 120,
                   'max_depth': 20,
                   'learning_rate': 0.02,
                   'gamma': 0.3,
                   'colsample_bytree': 0.5,
                   'random_state': 42}

    # initialize the model and parameters
    model = XGBRegressor(**params_dict)

    # fit model on training data
    model.fit(X_train, y_train)  
    # feature_importances = model.feature_importances_
    # print(feature_importances)
    
    # get predictions on test data
    y_pred = model.predict(X_test)

    return y_pred


def write_results(y_pred, rdd_test, output_file):
    # get just orginal ids
    ids_rdd = rdd_test.map(lambda x: (x[0], x[1])) 
    
    # print(ids_rdd.count())
    
    ids_rdd = ids_rdd.collect()

    # zip results
    results = [
        (user_id, business_id, pred)
        for (user_id, business_id), pred in zip(ids_rdd, y_pred)
    ]
    
    # print(len(results))
            
    with open(output_file, 'w', newline='') as f:
        header = "user_id, business_id, prediction"
        f.write(header + '\n')
        for row in results:
            f.write(','.join(map(str, row)) + '\n')
            
            
def rsme(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    print("RMSE:", rmse)
    
    # get error distribution
    abs_d = np.abs(y_pred - y_true)  # absolute differences

    # ranges for counts based on assignment
    ranges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, np.inf)]
    counts = {}

    for r in ranges:
        count = np.sum((abs_d >= r[0]) & (abs_d < r[1]))
        counts[f">={r[0]} and <{r[1]}" if r[1] != np.inf else f">={r[0]}"] = count

    for key, value in counts.items():
        print(f"{key}: n = {value}")


def main():
    start_time = time.time()
    rdd_train = add_features(folder_path, "yelp_train.csv", train=True)
    rdd_test = add_features(folder_path, test_file, train=False)
    # encoded_train, encoded_test = encode_ids(rdd_train, rdd_test)
    X_train, y_train, X_test = prep_features_target(rdd_train, rdd_test)
    y_pred = model_predictions(X_train, y_train, X_test)
    write_results(y_pred, rdd_test, output_file)
    duration = time.time() - start_time
    print("Duration: ", duration)
    # rsme(y_test, y_pred)


if __name__=="__main__":
    folder_path = str(sys.argv[1])
    test_file = str(sys.argv[2])
    output_file = str(sys.argv[3])
    main()
