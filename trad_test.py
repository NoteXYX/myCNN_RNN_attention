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
        kp_list = []
        cur_kp = []
        tmp = keywords_list.copy()
        con_index = 0
        while con_index < len(content_list) and len(tmp) > 0:
            key_index = 0
            while key_index < len(keywords_list):
                if content_list[con_index] == keywords_list[key_index]:
                    # cur_word = keywords_list[key_index]
                    cur_kp.append(keywords_list[key_index])
                    break
            if len(cur_kp) > 1 and con_index+1 < len(content_list):
                str_kp = ' '.join(cur_kp)
                kp_list.append(str_kp)
                for word in cur_kp:
                    delt = tmp.pop(tmp.index(word))
                cur_kp = []
            con_index += 1




        while len(tmp):
            kp_start = tmp.pop(key_index)
            for con_index in range(len(content_list)):
                cur_start = content_list[con_index]
                if cur_start == kp_start:
                    for i in range(con_index+1, len(content_list)):
                        if content_list[i] in tmp:




def main():


if __name__ == '__main__':
    main()
