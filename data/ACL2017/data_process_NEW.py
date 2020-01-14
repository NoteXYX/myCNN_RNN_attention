import numpy as np
import pickle
from collections import Counter
import gensim
import json
import nltk


def get_va_test_list(filename):
    datalist, taglist = [], []
    json_file = open(filename, 'r', encoding='utf-8')
    for line in json_file.readlines():
        json_data = json.loads(line)
        datalist.append(json_data["title"].strip().lower() + ' ' + json_data["abstract"].strip().lower())
        keywords_list = [keyword.strip() for keyword in json_data["keywords"].split(';')]
        keywords_str = '\t'.join(keywords_list)
        taglist.append(keywords_str)
    json_file.close()
    return datalist, taglist

def get_train_list(filename):
    datalist, taglist = [], []
    json_file = open(filename, 'r', encoding='utf-8')
    for line in json_file.readlines():
        json_data = json.loads(line)
        datalist.append(json_data["title"].strip().lower() + ' ' + json_data["abstract"].strip().lower())
        keywords_list = [keyword.strip() for keyword in json_data["keywords"]]
        keywords_str = '\t'.join(keywords_list)
        taglist.append(keywords_str)
    json_file.close()
    return datalist, taglist

# build vocabulary
def get_dict():
    kp20k_names = ['kp20k/kp20k_train.json', 'kp20k/kp20k_valid.json', 'kp20k/kp20k_test.json']
    inspec_names = ['inspec/inspec_valid.json', 'inspec/inspec_test.json']
    semeval_names = ['semeval/semeval_valid.json', 'semeval/semeval_test.json']
    nus_names = ['nus/nus_test.json']
    krapivin_names = ['krapivin/krapivin_valid.json', 'krapivin/krapivin_test.json']
    duc_names = ['duc/duc_test.json']
    train_kp20k, vaild_kp20k, test_kp20k = kp20k_names
    # vaild_inspec, test_inspec = inspec_names
    # vaild_semeval, test_semeval = semeval_names
    # test_nus = nus_names[0]
    # vaild_krapivin, test_krapivin = krapivin_names
    # test_duc = duc_names[0]
    names = kp20k_names[1:3] + inspec_names + semeval_names + nus_names + krapivin_names + duc_names
    sentence_list = get_train_list(train_kp20k)[0]
    i = 0
    for name in names:
        sentence_list += get_va_test_list(name)[0]
        i += 1
        if i % 10000 == 0:
            print('loaded %d sentences.........' % i)

    # sentence_list = get_va_test_list(train_f)[0] + get_va_test_list(vaild_f)[0] + get_va_test_list(test_f)[0]
    words = []
    i = 0
    for sentence in sentence_list:
        word_list = nltk.word_tokenize(sentence)
        words.extend(word_list)
        i += 1
        print('processed %d sentences.........' % i)
    word_counts = Counter(words)
    words2idx = {word[0]: i+1 for i, word in enumerate(word_counts.most_common())}
    idx2words = {v: k for (k, v) in words2idx.items()}
    labels2idx = {'O': 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}
    dicts = {'words2idx': words2idx, 'labels2idx': labels2idx, 'idx2words': idx2words}
    print('dict completed!')
    return dicts

def get_kp20k_train_valid_test_dicts(filenames, dataset_file, dicts):
    """
    Args:
    filenames:trnTweet,testTweet,tag_id_cnt

    Returns:
    dataset:train_set,test_set,dicts

    train_set=[train_lex,train_y,train_z]
    test_set=[test_lex,test_y,test_z]
    dicts = {'words2idx': words2idx, 'labels2idx': labels2idx}


    """
    train_doc, valid_doc, test_doc = filenames

    # trn_data = get_train_list(train_doc)
    trn_data = get_va_test_list(train_doc)
    valid_data = get_va_test_list(valid_doc)
    test_data = get_va_test_list(test_doc)

    trn_sentence_list, trn_tag_list = trn_data
    valid_sentence_list, valid_tag_list = valid_data
    test_sentence_list, test_tag_list = test_data

    words2idx = dicts['words2idx']
    labels2idx = dicts['labels2idx']

    def get_kp20k_lex_y(sentence_list, tag_list, words2idx):
        lex, y, z = [], [], []
        bad_num = 0
        num = 0
        for s, tag in zip(sentence_list, tag_list):
            num += 1
            word_list = nltk.word_tokenize(s)
            t_list = tag.split('\t')
            emb = list(map(lambda x: words2idx[x], word_list))
            all_keyphrase_sub = []
            for kp in t_list:
                win = len(kp.split(' '))
                for i in range(len(word_list)-win+1):
                    if ' '.join(word_list[i:i+win]) == kp:
                        cur_keyphrase_sub = list(range(i, i+win))
                        all_keyphrase_sub.append(cur_keyphrase_sub)
            cur_y = [0 for k in range(len(word_list))]
            cur_z = [0 for k in range(len(word_list))]
            if len(all_keyphrase_sub) > 0:
                for cur_sub in all_keyphrase_sub:
                    if len(cur_sub) == 1:
                        cur_y[cur_sub[0]] = 1
                        cur_z[cur_sub[0]] = labels2idx['S']
                    elif len(cur_sub) > 1:
                        cur_y[cur_sub[0]] = 1
                        cur_z[cur_sub[0]] = labels2idx['B']
                        for k in range(len(cur_sub) - 2):
                            cur_y[cur_sub[1 + k]] = 1
                            cur_z[cur_sub[1 + k]] = labels2idx['I']
                        cur_y[cur_sub[-1]] = 1
                        cur_z[cur_sub[-1]] = labels2idx['E']
                lex.append(emb)
                y.append(cur_y)
                z.append(cur_z)
            else:
                bad_num += 1
            if num % 1000 == 0:
                print('%d papers completed!' % num)
        print("Bad num = %d" % bad_num)
        return lex, y, z
    print('Bunilding train_lex, train_y, train_z...')
    train_lex, train_y, train_z = get_kp20k_lex_y(trn_sentence_list, trn_tag_list, words2idx)  # train_lex: [[每条tweet的word的idx],[每条tweet的word的idx]], train_y: [[关键词的位置为1]], train_z: [[关键词的位置为0~4(开头、结尾...)]]
    print('Bunilding valid_lex, valid_y, valid_z...')
    valid_lex, valid_y, valid_z = get_kp20k_lex_y(valid_sentence_list, valid_tag_list, words2idx)
    print('Bunilding test_lex, test_y, test_z...')
    test_lex, test_y, test_z = get_kp20k_lex_y(test_sentence_list, test_tag_list, words2idx)
    train_set = [train_lex, train_y, train_z]
    valid_set = [valid_lex, valid_y, valid_z]
    test_set = [test_lex, test_y, test_z]
    data_set = [train_set, valid_set, test_set, dicts]
    with open(dataset_file, 'wb') as f:
        pickle.dump(data_set, f)
    return data_set


def load_bin_vec(frame, vocab):
    k = 0
    word_vecs = {}
    model = gensim.models.KeyedVectors.load_word2vec_format(frame, binary=True)
    vec_vocab = model.vocab
    for word in vec_vocab:
        embedding = model[word]
        if word in vocab:
            word_vecs[word] = np.asarray(embedding, dtype=np.float32)
        k += 1
        if k % 10000 == 0:
            print("load_bin_vec %d" % k)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, dim=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    k = 0
    for w in vocab:
        if w not in word_vecs:
            word_vecs[w] = np.asarray(np.random.uniform(-0.25, 0.25, dim), dtype=np.float32)
            k += 1
            if k % 10000 == 0:
                print("add_unknow_words %d" % k)
    return word_vecs


def get_embedding(w2v, words2idx, emb_file, k=300):
    embedding = np.zeros((len(w2v) + 2, k), dtype=np.float32)
    for (w, idx) in words2idx.items():
        embedding[idx] = w2v[w]
    # embedding[0]=np.asarray(np.random.uniform(-0.25,0.25,k),dtype=np.float32)
    with open(emb_file, 'wb') as f:
        pickle.dump(embedding, f)
    return embedding


if __name__ == '__main__':
    kp20k_data_folder = ['kp20k/kp20k_train.json', 'kp20k/kp20k_valid.json', 'kp20k/kp20k_test.json']
    inspec_data_folder = ['inspec/inspec_valid.json', 'inspec/inspec_valid.json', 'inspec/inspec_test.json']
    semeval_data_folder = ['semeval/semeval_valid.json', 'semeval/semeval_valid.json', 'semeval/semeval_test.json']
    nus_data_folder = ['nus/split/nus_valid.json', 'nus/split/nus_valid.json', 'nus/nus_test.json']
    krapivin_data_folder = ['krapivin/krapivin_valid.json', 'krapivin/krapivin_valid.json', 'krapivin/krapivin_test.json']
    duc_data_folder = ['duc/split/duc_valid.json', 'duc/split/duc_valid.json', 'duc/duc_test.json']
    dicts = get_dict()
    vocab = set(dicts['words2idx'].keys())
    print("total num words: " + str(len(vocab)))

    kp20k_data_set = get_kp20k_train_valid_test_dicts(kp20k_data_folder, 'kp20k/kp20k_t_a_allwords_data_set.pkl', dicts)
    print("kp20k dataset created!")
    train_set, valid_set, test_set = kp20k_data_set[0:3]
    print("total kp20k train lines: " + str(len(train_set[0])))
    print("total kp20k valid lines: " + str(len(valid_set[0])))
    print("total kp20k test lines: " + str(len(test_set[0])))

    inspec_data_set = get_kp20k_train_valid_test_dicts(inspec_data_folder, 'inspec/inspec_t_a_allwords_data_set.pkl', dicts)
    print("inspec dataset created!")
    train_set, valid_set, test_set = inspec_data_set[0:3]
    print("total inspec train lines: " + str(len(train_set[0])))
    print("total inspec valid lines: " + str(len(valid_set[0])))
    print("total inspec test lines: " + str(len(test_set[0])))

    semeval_data_set = get_kp20k_train_valid_test_dicts(semeval_data_folder, 'semeval/semeval_t_a_allwords_data_set.pkl', dicts)
    print("semeval dataset created!")
    train_set, valid_set, test_set = semeval_data_set[0:3]
    print("total semeval train lines: " + str(len(train_set[0])))
    print("total semeval valid lines: " + str(len(valid_set[0])))
    print("total semeval test lines: " + str(len(test_set[0])))

    nus_data_set = get_kp20k_train_valid_test_dicts(nus_data_folder, 'nus/nus_t_a_allwords_data_set.pkl', dicts)
    print("nus dataset created!")
    train_set, valid_set, test_set = nus_data_set[0:3]
    print("total nus train lines: " + str(len(train_set[0])))
    print("total nus valid lines: " + str(len(valid_set[0])))
    print("total nus test lines: " + str(len(test_set[0])))

    krapivin_data_set = get_kp20k_train_valid_test_dicts(krapivin_data_folder, 'krapivin/krapivin_t_a_allwords_data_set.pkl', dicts)
    print("krapivin dataset created!")
    train_set, valid_set, test_set = krapivin_data_set[0:3]
    print("total krapivin train lines: " + str(len(train_set[0])))
    print("total krapivin valid lines: " + str(len(valid_set[0])))
    print("total krapivin test lines: " + str(len(test_set[0])))

    duc_data_set = get_kp20k_train_valid_test_dicts(duc_data_folder, 'duc/duc_t_a_allwords_data_set.pkl', dicts)
    print("duc dataset created!")
    train_set, valid_set, test_set = duc_data_set[0:3]
    print("total duc train lines: " + str(len(train_set[0])))
    print("total duc valid lines: " + str(len(valid_set[0])))
    print("total duc test lines: " + str(len(test_set[0])))

    # GoogleNews-vectors-negative300.txt为预先训练的词向量
    w2v_file = '../tweet_data/GoogleNews-vectors-negative300.bin'
    w2v = load_bin_vec(w2v_file, vocab)
    print ("word2vec loaded")
    w2v = add_unknown_words(w2v, vocab)
    embedding=get_embedding(w2v, dicts['words2idx'], 'ACL2017_t_a_embedding.pkl')
    print ("embedding created")
