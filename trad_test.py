import operator
import json
from rake import Rake

def rake_keyphrase(file_name, topk):
    json_file = open(file_name, 'r', encoding='utf-8')
    for line in json_file.readlines():
        json_data = json.loads(line)
        if file_name == 'data/ACL2017/kp20k/kp20k_train.json':
            golden_kp = [kp.strip() for kp in json_data['keywords']]
        else:
            golden_kp = [kp.strip() for kp in json_data['keywords'].split(';')]
        cur_content = json_data['title'].strip().lower() + ' ' + json_data['abstract'].strip().lower


def main():


if __name__ == '__main__':
    main()
