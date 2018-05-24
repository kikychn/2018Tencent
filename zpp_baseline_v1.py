import pandas as pd
from csv import DictWriter
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import lightgbm as lgb
import gc
import math
import numpy as np
import os

os.chdir(r'/home/zpp/PycharmProjects/2018TencentContest/data')

''' 将userFeature.data转换为userFeature.csv '''


def transform_user_feature_file():
    # 流式转换
    with open('userFeature.csv', 'w') as fo:
        headers = ['uid', 'age', 'gender', 'marriageStatus', 'education',
                   'consumptionAbility', 'LBS', 'interest1',
                   'interest2', 'interest3', 'interest4', 'interest5', 'kw1',
                   'kw2', 'kw3', 'topic1', 'topic2',
                   'topic3', 'appIdInstall', 'appIdAction', 'ct', 'os',
                   'carrier', 'house']
        writer = DictWriter(fo, fieldnames=headers, lineterminator='\n')
        writer.writeheader()

        fi = open('userFeature.data', 'r')
        for i, line in enumerate(fi):
            line = line.strip().replace('\n', '').split('|')
            userFeature_dict = {}
            for each in line:
                each_list = each.split(' ')
                userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
            writer.writerow(userFeature_dict)
            if i % 100000 == 0:
                print(i)
        fi.close()
    print('Transform userFeature.data to  userFeature.csv successfully.')


def get_user_feature():
    if not os.path.exists('userFeature.csv'):
        transform_user_feature_file()

    user_feature = pd.read_csv('userFeature.csv')

    # reader = pd.read_csv('userFeature.csv',
    #                      iterator=True)  # userFeature是可迭代对象TextFileReader
    # chunkSize = 10000
    # user_feature = reader.get_chunk(chunkSize)

    print('Load userFeature successfully.')
    return user_feature


def get_ad_feature():
    ad_feature = pd.read_csv('adFeature.csv')

    # reader = pd.read_csv('adFeature.csv',
    #                      iterator=True)  # userFeature是可迭代对象TextFileReader
    # chunkSize = 10000
    # ad_feature = reader.get_chunk(chunkSize)

    print('Load adFeature successfully.')
    return ad_feature


def get_train():
    train = pd.read_csv('train.csv')

    # reader = pd.read_csv('train.csv',
    #                      iterator=True)  # userFeature是可迭代对象TextFileReader
    # chunkSize = 10000
    # train = reader.get_chunk(chunkSize)

    print('Load train data successfully.')
    return train


def get_test():
    test = pd.read_csv('test1.csv')

    # reader = pd.read_csv('test1.csv',
    #                      iterator=True)  # userFeature是可迭代对象TextFileReader
    # chunkSize = 10000
    # test = reader.get_chunk(chunkSize)

    print('Load test data successfully.')
    return test


def get_data():
    # if os.path.exists('data.csv'):
    #     return pd.read_csv('data.csv')
    # else:

    train = get_train()
    test = get_test()
    ad_feature = get_ad_feature()
    user_feature = get_user_feature()

    train.loc[train['label'] == -1, 'label'] = 0
    test['label'] = -1
    data = pd.concat([train, test])
    data = pd.merge(data, ad_feature, on='aid', how='left')
    data = pd.merge(data, user_feature, on='uid', how='left')
    data = data.fillna('-1')
    del user_feature, ad_feature, train, test
    # data.to_csv('data.csv')
    return data


def feature_engineering(data):
    one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility',
                       'education', 'gender', 'house', 'os', 'ct',
                       'marriageStatus', 'advertiserId', 'campaignId',
                       'creativeId', 'adCategoryId', 'productId', 'productType']
    vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2',
                      'interest3', 'interest4', 'interest5',
                      'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
    for feature in one_hot_feature:
        try:
            data[feature] = LabelEncoder().fit_transform(
                data[feature].apply(int))
        except:
            data[feature] = LabelEncoder().fit_transform(data[feature])

    train = data[data.label != -1]
    train_y = train.pop('label')
    test = data[data.label == -1]
    res = test[['aid', 'uid']]
    test = test.drop('label', axis=1)
    train_x = train[['creativeSize']]
    test_x = test[['creativeSize']]
    # train_x = train
    # test_x = test



    enc = OneHotEncoder()
    for feature in one_hot_feature:
        enc.fit(data[feature].values.reshape(-1, 1))
        # train[feature].values.reshape(-1, 1)将训练集中的某个特征列如['house', '-1', '1']转换成[[house],[-1],[1]]
        train_a = enc.transform(train[feature].values.reshape(-1, 1))
        test_a = enc.transform(test[feature].values.reshape(-1, 1))
        train_x = sparse.hstack((train_x, train_a))  # hstack两个矩阵水平拼接起来
        test_x = sparse.hstack((test_x, test_a))
        print(feature + ' finish')
    print('one_hot_feature prepared !')

    '''将某向量特征列表示为每种类别出现次数的向量
    如intreste1中所有类别为[1,3,5,7,20],对应一条记录的向量表示为[0,1,0,1,0]，
    则表明该条记录中3和7各出现一次'''
    cv = CountVectorizer(token_pattern='(?u)\\b\\w+\\b')
    # cv = CountVectorizer()
    for feature in vector_feature:
        cv.fit(data[feature])
        train_a = cv.transform(train[feature])
        test_a = cv.transform(test[feature])
        train_x = sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))
        print(feature + ' finish')
    print('vector_feature prepared !')

    return train_x, train_y, test_x, res


def LGB_predict(train_x, train_y, test_x, res, index):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1, max_depth=-1, n_estimators=1000, objective='binary', subsample=0.7, colsample_bytree=0.7, subsample_freq=1, learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc', early_stopping_rounds=100)
    res['score' + str(index)] = clf.predict_proba(test_x)[:, 1]
    res['score' + str(index)] = res['score' + str(index)].apply(
        lambda x: float('%.6f' % x))
    print(str(index) + ' predict finish!')
    gc.collect()
    res = res.reset_index(drop=True)
    return res['score' + str(index)]


def batch_predict(data, index):
    train_x, train_y, test_x, res = feature_engineering(data)
    return LGB_predict(train_x, train_y, test_x, res, index)


data = get_data()
train = data[data['label'] != -1]
test = data[data['label'] == -1]
del data
cnt = 20
size = math.ceil(len(train) / cnt)
result = []
for i in range(cnt):
    start = size * i
    end = (i + 1) * size if (i + 1) * size < len(train) else len(train)
    slice = train[start:end]
    result.append(batch_predict(pd.concat([slice, test]), i))
    gc.collect()

predict = pd.read_csv('test1.csv')
result = pd.concat(result, axis=1)
result['score'] = np.mean(result, axis=1)
result = result.reset_index(drop=True)
result = pd.concat(
    [predict[['aid', 'uid']].reset_index(drop=True), result['score']], axis=1)
result[['aid', 'uid', 'score']].to_csv('submission.csv', index=False)
