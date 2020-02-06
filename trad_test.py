import operator
import nltk
import json
from rake import Rake

def rake_keyphrase(file_name, topk):
    json_file = open(file_name, 'r', encoding='utf-8')
    for line in json_file.readlines():
        json_data = json.loads(line)
        if file_name == 'data/ACL2017/kp20k/kp20k_train.json':
            golden_list = [kp.strip() for kp in json_data['keywords']]
        else:
            golden_list = [kp.strip() for kp in json_data['keywords'].split(';')]
        cur_content = json_data['title'].strip().lower() + ' ' + json_data['abstract'].strip().lower
        content_list = nltk.word_tokenize(cur_content)
        rake = Rake()
        keywords_dict = rake.run(cur_content)
        keywords_list = list(keywords_dict.keys())[:topk]
        for key_index in range(len(keywords_list)):
            tmp = keywords_list.copy()
            kp_start = tmp.pop(key_index)
            for con_index in range(len(content_list)):
                cur_start = content_list[con_index]
                if cur_start == kp_start:
                    for i in range(content_list+1, len(content_list)):
                        if content_list[i] in tmp:









def main():


if __name__ == '__main__':
    main()
