import matplotlib.pyplot as plt

def feature_analyze(data):
    feature_distribution(data)


def feature_distribution(data):
    figNum = 1
    fig = plt.figure('各属性分布图'+str(figNum))
    fig.set(alpha=0.2)
    idx = 1

    plt.subplot(230 + idx)
    idx += 1
    data.label.value_counts().plot(kind='bar')
    plt.title('正负样本分布')
    plt.ylabel('样本数量')

    plt.subplot(230 + idx)
    idx += 1
    data.age.value_counts().plot(kind='bar')
    plt.title('年龄分布')
    plt.ylabel('样本数量')

    plt.subplot(230 + idx)
    idx += 1
    data.gender.value_counts().plot(kind='bar')
    plt.title('性别分布')
    plt.ylabel('样本数量')

    plt.subplot(230 + idx)
    idx += 1
    data.marriageStatus.value_counts().plot(kind='bar')
    plt.title('婚姻状况分布')
    plt.ylabel('样本数量')

    plt.subplot(230 + idx)
    idx += 1
    data.education.value_counts().plot(kind='bar')
    plt.title('学历分布')
    plt.ylabel('样本数量')

    plt.subplot(230 + idx)
    idx += 1
    data.consumptionAbility.value_counts().plot(kind='bar')
    plt.title('消费能力分布')
    plt.ylabel('样本数量')

    plt.show()


    figNum += 1
    fig = plt.figure('各属性分布图' + str(figNum))
    fig.set(alpha=0.2)
    idx = 1

    plt.subplot(230 + idx)
    idx += 1
    data.LBS.value_counts().plot(kind='bar')
    plt.title('地理位置分布')
    plt.ylabel('样本数量')

    # plt.subplot(230 + idx)
    # idx += 1
    # data.interest1.value_counts().plot(kind='bar')
    # plt.title('兴趣1分布')
    # plt.ylabel('样本数量')
    #
    # plt.subplot(230 + idx)
    # idx += 1
    # data.interest2.value_counts().plot(kind='bar')
    # plt.title('兴趣2分布')
    # plt.ylabel('样本数量')
    #
    # plt.subplot(230 + idx)
    # idx += 1
    # data.interest3.value_counts().plot(kind='bar')
    # plt.title('兴趣3分布')
    # plt.ylabel('样本数量')
    #
    # plt.subplot(230 + idx)
    # idx += 1
    # data.interest4.value_counts().plot(kind='bar')
    # plt.title('兴趣4分布')
    # plt.ylabel('样本数量')
    #
    # plt.subplot(230 + idx)
    # idx += 1
    # data.interest5.value_counts().plot(kind='bar')
    # plt.title('兴趣5分布')
    # plt.ylabel('样本数量')

    # plt.subplot(230 + idx)
    # idx += 1
    # data.age[data.education == 0].plot(kind='kde')
    # data.age[data.education == 1].plot(kind='kde')
    # data.age[data.education == 2].plot(kind='kde')
    # plt.title('各学历的年龄分布')
    # plt.xlabel('年龄')
    # plt.ylabel('密度')

    plt.show()