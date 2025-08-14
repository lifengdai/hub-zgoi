import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# 读取dataset.csv并保存至dataset中
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)

# 提取 文本的特征 tfidf， dataset[0]
# 构建一个模型 knn， 学习 提取的特征和 标签 dataset[1] 的关系
# 预测，用户输入的一个文本，进行预测结果
input_sententce = dataset[0].apply(lambda x: " ".join(jieba.cut(x))) # sklearn对中文处理

vector = CountVectorizer() # 对文本进行提取特征 默认是使用标点符号分词
vector.fit(input_sententce.values)
input_feature = vector.transform(input_sententce.values)

model1 = KNeighborsClassifier()
model1.fit(input_feature, dataset[1].values)

model2 = MultinomialNB()
model2.fit(input_feature, dataset[1].values)

model3 = RandomForestClassifier()
model3.fit(input_feature, dataset[1].values)

test_query = "帮我播放一下郭德纲的小品"
test_sentence = " ".join(jieba.cut(test_query))
test_feature = vector.transform([test_sentence])

print("待预测的文本", test_query)
print("KNN模型预测结果: ", model1.predict(test_feature))
print("贝叶斯: ", model2.predict(test_feature))
print("随机森林: ", model3.predict(test_feature))

# 新词发现算法： 按照字之间的联合出现的频次，发现新的成语
#  -> 特定的场景
# subword token