import operator
import nltk
import json
from rake import Rake
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


def get_kp(content_list, keywords_list):
    kp_list = []
    cur_kp = []
    tmp = keywords_list.copy()
    con_index = 0
    while con_index < len(content_list) and len(tmp) > 0:
        if content_list[con_index] in tmp:
            cur_kp.append(content_list[con_index])
            con_index += 1
            if con_index == len(content_list) and len(cur_kp) > 1:
                str_kp = ' '.join(cur_kp)
                kp_list.append(str_kp)
            continue
        elif len(cur_kp) > 1:
            str_kp = ' '.join(cur_kp)
            kp_list.append(str_kp)
            for word in cur_kp:
                delt = tmp.pop(tmp.index(word))
        cur_kp = []
        con_index += 1
    kp_list = keywords_list.extend(kp_list)
    return kp_list

def get_rake_kp(file_name, topk):
    json_file = open(file_name, 'r', encoding='utf-8')
    rake_kp = []
    for line in json_file.readlines():
        json_data = json.loads(line)
        cur_content = json_data['title'].strip().lower() + ' ' + json_data['abstract'].strip().lower
        content_list = nltk.word_tokenize(cur_content)
        rake = Rake()
        keywords_dict = rake.run(cur_content)
        keywords_list = list(keywords_dict.keys())[:topk]
        kp_list = get_kp(content_list, keywords_list)
        rake_kp.append(kp_list)
    json_file.close()
    return rake_kp

def get_tfidf_kp(file_name, topk):
    json_file = open(file_name, 'r', encoding='utf-8')
    tfidf_kp = []
    corpus = []
    for line in json_file.readlines():
        json_data = json.loads(line)
        cur_content = json_data['title'].strip().lower() + ' ' + json_data['abstract'].strip().lower
        corpus.append(cur_content)
    json_file.close()
    vectorizer = CountVectorizer(lowercase=True)  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    for line_index in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
        weight_index = dict()
        for word_index in range(len(word)):
            weight_index[weight[line_index][word_index]] = word_index
        sorted_weight_index = dict(
            sorted(weight_index.items(), key=operator.itemgetter(0), reverse=True)[0: min(topn, len(word)): 1])
        line_keywords = list()
        for keyword_weight in sorted_weight_index:
            line_keywords.append(word[sorted_weight_index[keyword_weight]])
            # print(word[sorted_weight_index[keyword_weight]], keyword_weight)
        tfidf_keywords[line_index] = line_keywords

def mytfidf(test_name, stopwords, keywordstop, topn=20):  # TF-IDF算法，返回字典{index:[关键字1,关键字2...]}
    corpus = list()
    tfidf_keywords = dict()
    with open(test_name, 'r', encoding='utf-8') as test_file:
        num = 0
        for test_line in test_file.readlines():
            line_split = test_line.split(' ::  ')
            if len(line_split) == 2:
                content = line_split[1].strip()
                print('第%d条专利摘要：' % (num + 1))
                print(content)
                test_line_words = list(jieba.cut(content))
                line_words = list()
                for word in test_line_words:
                    if word not in stopwords and word not in keywordstop and len(word) > 1 and not word.isdigit():
                        line_words.append(word)
                line_str = ' '.join(line_words)
                corpus.append(line_str)
                num += 1
    vectorizer = CountVectorizer(lowercase=False)  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(
        vectorizer.fit_transform(corpus))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    for line_index in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
        weight_index = dict()
        for word_index in range(len(word)):
            weight_index[weight[line_index][word_index]] = word_index
        sorted_weight_index = dict(
            sorted(weight_index.items(), key=operator.itemgetter(0), reverse=True)[0: min(topn, len(word)): 1])
        line_keywords = list()
        for keyword_weight in sorted_weight_index:
            line_keywords.append(word[sorted_weight_index[keyword_weight]])
            # print(word[sorted_weight_index[keyword_weight]], keyword_weight)
        tfidf_keywords[line_index] = line_keywords
    return tfidf_keywords

def get_golden_kp(file_name):
    golden_kp = []
    json_file = open(file_name, 'r', encoding='utf-8')
    for line in json_file.readlines():
        json_data = json.loads(line)
        if file_name == 'data/ACL2017/kp20k/kp20k_train.json':
            golden_list = [kp.strip().lower() for kp in json_data['keywords']]
        else:
            golden_list = [kp.strip().lower() for kp in json_data['keywords'].split(';')]
        golden_kp.append(golden_list)
    json_file.close()
    return golden_kp





def main():


if __name__ == '__main__':
    main()
