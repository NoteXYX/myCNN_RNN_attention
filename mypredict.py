import tensorflow as tf
import load
import models.mymodel_CNN_ori as mymodel
import models.mymodel_multi_CNN as mymodelmultiCNN
import models.mymodel_multi_CNN_NEW as mymodelmultiCNN_NEW
import models.mymodel_CNN_LSTM as mymodel1
import tools

def batch_putin(train, test, start_num=0, batch_size=16):
    batch = [train[start_num:start_num + batch_size], test[start_num:start_num + batch_size]]
    return batch


def main():
    s = {
        # 'nh1': 300,
        # 'nh2': 300,
        'nh1': 400,
        'nh2': 400,
        'win': 3,
        'emb_dimension': 300,
        'lr': 0.1,
        'lr_decay': 0.5,
        'max_grad_norm': 5,
        'seed': 345,
        'nepochs': 50,
        'batch_size': 16,
        'keep_prob': 1.0,
        'check_dir': './mycheckpoints_multi_CNN_NEW',
        'display_test_per': 5,
        'lr_decay_per': 10
    }

    # load the dataset
    train_set, test_set, dic, embedding = load.atisfold_old()
    # idx2label = dict((k, v) for v, k in dic['labels2idx'].items())
    # idx2word = dict((k, v) for v, k in dic['words2idx'].items())

    # vocab = set(dic['words2idx'].keys())
    # vocsize = len(vocab)
    logfile = open(str(s['check_dir']) + '/predict_log.txt', 'w', encoding='utf-8')
    test_lex, test_y, test_z = test_set
    #test_lex = test_lex[:1000]
    #test_y = test_y[:1000]
    #test_z = test_z[:1000]

    y_nclasses = 2
    z_nclasses = 5

    with tf.Session() as sess:
        my_model = mymodelmultiCNN_NEW.myModel(
            nh1=s['nh1'],
            nh2=s['nh2'],
            ny=y_nclasses,
            nz=z_nclasses,
            de=s['emb_dimension'],
            lr=s['lr'],
            lr_decay=s['lr_decay'],
            embedding=embedding,
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

        def dev_step(cwords):
            feed = {
                my_model.cnn_input_x: cwords,
                # my_model.rnn_ori_input_x: cwords
                # rnn.keep_prob:1.0,
                # rnn.batch_size:s['batch_size']
            }
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
            batch = batch_putin(test_lex, test_z, start_num=start_num, batch_size=s['batch_size'])
            x, z = batch
            x = load.pad_sentences(x)
            # x = tools.contextwin_2(x, s['win'])
            predictions_test.extend(dev_step(x))
            groundtruth_test.extend(z)
            start_num += s['batch_size']

        res_test = tools.conlleval(predictions_test, groundtruth_test, '')

        print(res_test)
        logfile.write(str(res_test))
    logfile.close()

if __name__ == '__main__':
    main()




