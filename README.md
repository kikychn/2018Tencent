# 2018腾讯广告算法大赛

网址：http://algo.tpai.qq.com/

赛题：http://algo.tpai.qq.com/home/information/info.html

数据下载链接: https://pan.baidu.com/s/1v98iSugDEG9OJkJg1rVHEw 密码: nwjj


zpp_baseline_v1.py对应初赛Ａ阶段的test1数据集；
zpp_baseline_v２.1.py对应初赛B阶段的test2数据集；
两阶段评分均为0.71。

流程简介：
1. 将train/adFeature/userFeature通过aid和uid进行拼接，存入data.csv。
2. 特征转换，单值转换为one-hot形式，多值转换为频数向量形式。
3. 调用lightgbm进行分类，结果存入submission.csv。

编程语言：Python3.6(使用Anaconda3)

运行环境：内存32ｇ的服务器