import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import lightgbm as lgb
import gc
import math
import numpy as np
import os

os.chdir(r'/home/zpp/PycharmProjects/2018TencentContest/data')

if not os.path.exists('data_v2.1.csv'):
    # 加载adFeature
    print('adFeature loading...')
    ad_feature = pd.read_csv('adFeature.csv')
    print('Load adFeature successfully.')

    # 加载userFeature
    if not os.path.exists('userFeature_v2.1.csv'):
        print('Transform userFeature.data to  userFeature_v2.1.csv starting...')
        userFeature_data = []
        with open('userFeature.data', 'r') as f:
            for i, line in enumerate(f):
                line = line.strip().split('|')
                userFeature_dict = {}
                for each in line:
                    each_list = each.split(' ')
                    userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
                userFeature_data.append(userFeature_dict)
                if i % 100000 == 0:
                    print(i)
            user_feature = pd.DataFrame(userFeature_data)
            user_feature.to_csv('userFeature_v2.1.csv', index=False)
        print(
            'Transform userFeature.data to  userFeature_v2.1.csv successfully.')
    else:
        print('userFeature_v2.1 loading...')
        user_feature = pd.read_csv('userFeature_v2.1.csv')
        print('Load userFeature successfully.')

    # 加载train
    print('train loading...')
    train = pd.read_csv('train.csv')
    print('Load train data successfully.')

    # 加载test2
    print('test2 loading...')
    test = pd.read_csv('test2.csv')
    print('Load test data successfully.')

    # 拼接train/test/adFeature/userFeature
    print('train/test/adFeature/userFeature merging...')
    train.loc[train['label'] == -1, 'label'] = 0
    test['label'] = -1
    data = pd.concat([train, test])
    data = pd.merge(data, ad_feature, on='aid', how='left')
    data = pd.merge(data, user_feature, on='uid', how='left')
    data = data.fillna('-1')
    del user_feature, ad_feature, train, test
    print('merge finish.')
    print('data_v2.1.csv writing...')
    data.to_csv('data_v2.1.csv')
    print('data_v2.1.csv is writen finished!')

else:
    # 加载拼接后的数据data
    print('data_v2.1 loading...')
    data = pd.read_csv('data_v2.1.csv')
    print('Load data_v2.1 successfully.')

# 特征工程
def feature_engineering(data):
    print('feature_engineering starting...')
    one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility',
                       'education', 'gender', 'house', 'os', 'ct',
                       'marriageStatus', 'advertiserId', 'campaignId',
                       'creativeId', 'adCategoryId', 'productId', 'productType']
    vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2',
                      'interest3', 'interest4', 'interest5',
                      'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']

    print('  one_hot_feature transform to int...')
    #　将one-hot特征的数据类型转换为int64
    for feature in one_hot_feature:
        try:
            data[feature] = LabelEncoder().fit_transform(
                data[feature].apply(int))
        except:
            data[feature] = LabelEncoder().fit_transform(data[feature])
    print('  transform finish.')

    print('  train/train_y/test/res/train_x/test_x building...')
    train = data[data.label != -1]
    train_y = train.pop('label')
    test = data[data.label == -1]
    res = test[['aid', 'uid']]
    test = test.drop('label', axis=1)
    train_x = train[['creativeSize']]
    test_x = test[['creativeSize']]
    print('  build finish.')

    print('  one_hot_feature transform starting...')
    enc = OneHotEncoder()
    for feature in one_hot_feature:
        enc.fit(data[feature].values.reshape(-1, 1))
        # train[feature].values.reshape(-1, 1)将训练集中的某个特征列如['house', '-1', '1']转换成[[house],[-1],[1]]
        train_a = enc.transform(train[feature].values.reshape(-1, 1))
        test_a = enc.transform(test[feature].values.reshape(-1, 1))
        train_x = sparse.hstack((train_x, train_a))  # hstack两个矩阵水平拼接起来
        test_x = sparse.hstack((test_x, test_a))
        del train_a, test_a
        print('    ' + feature + ' finish')
    print('  one_hot_feature prepared !')

    '''将某向量特征列表示为每种类别出现次数的向量
    如intreste1中所有类别为[1,3,5,7,20],对应一条记录的向量表示为[0,1,0,1,0]，
    则表明该条记录中3和7各出现一次'''
    print('  vector_feature transform starting...')
    # cv = CountVectorizer(token_pattern='(?u)\\b\\w+\\b')
    cv = CountVectorizer()
    for feature in vector_feature:
        cv.fit(data[feature])
        train_a = cv.transform(train[feature])
        test_a = cv.transform(test[feature])
        train_x = sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))
        print('    ' + feature + ' finish')
    print('  vector_feature prepared !')

    print('feature_engineering finish.')

    return train_x, train_y, test_x, res


def lgb_predict(train_x, train_y, test_x, res, index):
    from multiprocessing import cpu_count
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=1500, objective='binary', subsample=0.7,
        colsample_bytree=0.7, subsample_freq=1, learning_rate=0.05,
        min_child_weight=50, random_state=2018, n_jobs=cpu_count() - 1
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc',
            early_stopping_rounds=100)
    # print(clf.feature_importances_)
    # return clf, clf.best_score_['valid_1']['auc']
    res['score' + str(index)] = clf.predict_proba(test_x)[:, 1]
    res['score' + str(index)] = res['score' + str(index)].apply(
        lambda x: float('%.6f' % x))
    print(str(index) + ' predict finish!')
    gc.collect()
    res = res.reset_index(drop=True)
    return res['score' + str(index)]


def batch_predict(data, index):
    train_x, train_y, test_x, res = feature_engineering(data)
    return lgb_predict(train_x, train_y, test_x, res, index)

train = data[data['label'] != -1]
test = data[data['label'] == -1]
del data
cnt = 10
size = math.ceil(len(train) / cnt)
result = []
for i in range(cnt):
    start = size * i
    end = (i + 1) * size if (i + 1) * size < len(train) else len(train)
    slice = train[start:end]
    result.append(batch_predict(pd.concat([slice, test]), i))
    gc.collect()

predict = pd.read_csv('test2.csv')
result = pd.concat(result, axis=1)
result['score'] = np.mean(result, axis=1)
result = result.reset_index(drop=True)
result = pd.concat(
    [predict[['aid', 'uid']].reset_index(drop=True), result['score']],
    axis=1)

print('store res to submission_v2.1.csv')
result[['aid', 'uid', 'score']].to_csv('submission.csv', index=False)
print('store finish.')
print('zip submission_v2.1.csv')
os.system('zip submission_v2.1.zip submission.csv')
print('zip finish.')

print('predict finish!')