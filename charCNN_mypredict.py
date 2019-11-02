import tensorflow as tf
import load
import models.charCNN_LSTM_attention as charCNN_LSTM_attention
import tools

def batch_putin(train, test=None, start_num=0, batch_size=16):
    if test is None:
        batch = train[start_num:start_num + batch_size]
    else:
        batch = [train[start_num:start_num + batch_size], test[start_num:start_num + batch_size]]
    return batch

def get_charidx(lex, idx2word, char2idx):
    char_idx = []
    for wordidx_list in lex:
        sentence_charidx = []
        for wordidx in wordidx_list:
            word_charidx = []
            cur_word = idx2word[wordidx]
            for char in cur_word:
                cur_charidx = char2idx[char]
                word_charidx.append(cur_charidx)
            sentence_charidx.append(word_charidx)
        char_idx.append(sentence_charidx)
    return char_idx     # (文本数，单词数，字母数)

def main():
    s = {
        'nh1': 330,
        'nh2': 330,
        'word_emb_dimension': 300,
        'lr': 0.1,
        'lr_decay': 0.5,  #
        'max_grad_norm': 5,  #
        'seed': 345,  #
        'nepochs': 50,
        'batch_size': 16,
        'keep_prob': 0.5,
        'check_dir': './mycheckpoints_charCNN_LSTM_attention',
        'display_test_per': 3,  #
        'lr_decay_per': 6  #
    }

    # load the dataset
    train_set, test_set, dic, word_embedding, idx2word, char_embedding, char2idx = load.atisfold()
    # idx2label = dict((k, v) for v, k in dic['labels2idx'].items())
    # idx2word = dict((k, v) for v, k in dic['words2idx'].items())

    # vocab = set(dic['words2idx'].keys())
    # vocsize = len(vocab)
    logfile = open(str(s['check_dir']) + '/predict_log.txt', 'w', encoding='utf-8')
    test_lex, test_y, test_z = test_set
    test_lex = test_lex[:1000]
    test_y = test_y[:1000]
    test_z = test_z[:1000]
    test_char_lex = get_charidx(test_lex, idx2word, char2idx)

    y_nclasses = 2
    z_nclasses = 5

    with tf.Session() as sess:
        my_model = charCNN_LSTM_attention.myModel(
            nh1=s['nh1'],
            nh2=s['nh2'],
            ny=y_nclasses,
            nz=z_nclasses,
            de=s['word_emb_dimension'],
            lr=s['lr'],
            lr_decay=s['lr_decay'],
            word_embedding=word_embedding,
            char_embedding=char_embedding,
            max_gradient_norm=s['max_grad_norm'],
            keep_prob=s['keep_prob'],
            rnn_model_cell='lstm'

        )


        checkpoint_dir = s['check_dir']
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # print(ckpt.all_model_checkpoint_paths[4])
            print(ckpt.model_checkpoint_path)
            logfile.write(str(ckpt.model_checkpoint_path) + '\n')
            saver.restore(sess, ckpt.model_checkpoint_path)

        def dev_step(word_input_x, char_input_x):
            feed = {
                my_model.input_word_idx: word_input_x,
                my_model.input_char_idx: char_input_x,
            }
            my_model.keep_prob = 1.0
            fetches = my_model.sz_pred
            sz_pred = sess.run(fetches=fetches, feed_dict=feed)
            return sz_pred

        print("测试结果：")
        logfile.write("测试结果：\n")
        predictions_test = []
        groundtruth_test = []
        start_num = 0
        steps = len(test_lex) // s['batch_size']
        # for batch in tl.iterate.minibatches(test_lex, test_z, batch_size=s['batch_size']):
        for step in range(steps):
            batch = batch_putin(test_lex, test=test_z, start_num=start_num, batch_size=s['batch_size'])
            char_input_x = batch_putin(test_char_lex, test=None, start_num=start_num, batch_size=s['batch_size'])
            x, z = batch
            x = load.pad_sentences(x)
            char_input_x = load.pad_chars(char_input_x)
            predictions_test.extend(dev_step(x, char_input_x))
            groundtruth_test.extend(z)
            start_num += s['batch_size']

        res_test = tools.conlleval(predictions_test, groundtruth_test, '')

        print(res_test)
        logfile.write(str(res_test))
    logfile.close()

if __name__ == '__main__':
    main()




