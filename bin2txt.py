import gensim
import codecs


def main():
    path_to_model = 'D:\PycharmProjects\myCNN_RNN_attention\data\GoogleNews-vectors-negative300.bin'
    output_file = 'D:\PycharmProjects\myCNN_RNN_attention\data\GoogleNews-vectors-negative300.txt'
    bin2txt(path_to_model, output_file)


def bin2txt(path_to_model, output_file):
    output = open(output_file, 'w', encoding='utf-8')
    model = gensim.models.KeyedVectors.load_word2vec_format(path_to_model, binary=True)
    print('Done loading Word2Vec!')
    vocab = model.vocab
    for item in vocab:
        vector = list()
        for dimension in model[item]:
            vector.append(str(dimension))
        vector_str = " ".join(vector)
        line = item + "\t" + vector_str
        output.write(line + "\n")  # 本来用的是write（）方法，但是结果出来换行效果不对。改成writelines（）方法后还没试过。
    output.close()


if __name__ == "__main__":
    main()
