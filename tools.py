# -*- coding: utf-8 -*-
import random

def shuffle(lol,seed):
    '''
    lol :: list of list as input
    seed :: seed the shuffling

    shuffle inplace each list in the same order
    '''
    for l in lol:
        random.seed(seed)
        random.shuffle(l)

def contextwin(l, win):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >=1
    l = list(l)

    lpadded = win//2 * [0] + l + win//2 * [0]
    out = [ lpadded[i:i+win] for i in range(len(l)) ]

    assert len(out) == len(l)
    return out

def contextwin_2(ls,win):
    assert (win % 2) == 1
    assert win >=1
    outs=[]
    for l in ls:
        outs.append(contextwin(l,win))
    return outs

def getKeyphraseList(l):
    res, now= [], []
    singleKW = []
    moreKP = []
    for i in range(len(l)):
        if l[i] != 0:
            now.append(str(i))
        if l[i] == 0 or i == len(l) - 1:
            if len(now) != 0:
                res.append(' '.join(now))
                if len(now) == 1:
                    singleKW.append(now[0])
                else:
                    moreKP.append(' '.join(now))
            now = []
    return set(res), set(singleKW), set(moreKP)

def getKeyphraseList_top(l, top_num):
    res, now= [], []
    singleKW = []
    moreKP = []
    for i in range(len(l)):
        if l[i] != 0:
            now.append(str(i))
        if l[i] == 0 or i == len(l) - 1:
            if len(now) != 0:
                res.append(' '.join(now))
                if len(now) == 1:
                    singleKW.append(now[0])
                else:
                    moreKP.append(' '.join(now))
                if len(res) >= top_num:
                    break
            now = []
    return set(res), set(singleKW), set(moreKP)

def conlleval(predictions, groundtruth):
    assert len(predictions) == len(groundtruth)
    res = {}
    all_cnt, good_cnt = len(predictions), 0
    p_cnt, r_cnt = 0, 0
    goodSingleCnt, goodMoreCnt = 0, 0
    singlePreCnt, singleRecallCnt = 0, 0
    morePreCnt, moreRecallCnt = 0, 0
    for i in range(all_cnt):
        # print i
        # if all(predictions[i][0:len(groundtruth[i])] == groundtruth[i]) == True:
        #     good_cnt += 1
        pKeyphraseList, pSingleKpList, pMoreKpList = getKeyphraseList(predictions[i][0:len(groundtruth[i])])
        gKeyphraseList, gSingleKpList, gMoreKpList = getKeyphraseList(groundtruth[i])
        for p in pKeyphraseList:
            for g in gKeyphraseList:
                if p == g:
                    good_cnt += 1
                    break
        for p in pSingleKpList:
            for g in gSingleKpList:
                if p == g:
                    goodSingleCnt += 1
                    break
        for p in pMoreKpList:
            for g in gMoreKpList:
                if p == g:
                    goodMoreCnt += 1
                    break
        p_cnt += len(pKeyphraseList)  #######################
        r_cnt += len(gKeyphraseList)  #######################
        singlePreCnt += len(pSingleKpList)
        singleRecallCnt += len(gSingleKpList)
        morePreCnt += len(pMoreKpList)
        moreRecallCnt += len(gMoreKpList)
        # if len(pKeyphraseList) != 0:
        #     p_cnt += 1
        # if len(gKeyphraseList) != 0:
        #     r_cnt += 1
        # pr_cnt += len(pKeyphraseList & gKeyphraseList)
    res['a'] = 1.0*good_cnt/all_cnt
    res['p'] = 1.0*good_cnt/p_cnt
    res['r'] = 1.0*good_cnt/r_cnt
    res['f'] = 2.0*res['p']*res['r']/(res['p']+res['r'])
    res['SinglePre'] = 1.0 * goodSingleCnt/singlePreCnt
    res['SingleRecall'] = 1.0 * goodSingleCnt/singleRecallCnt
    res['SingleF1'] = 2.0 * res['SinglePre']*res['SingleRecall']/(res['SinglePre']+res['SingleRecall'])
    res['MorePre'] = 1.0 * goodMoreCnt / morePreCnt
    res['MoreRecall'] = 1.0 * goodMoreCnt / moreRecallCnt
    res['MoreF1'] = 2.0 * res['MorePre'] * res['MoreRecall'] / (res['MorePre'] + res['MoreRecall'])
    return res

def conlleval_top(predictions, groundtruth, top_num):
    assert len(predictions) == len(groundtruth)
    res = {}
    all_cnt, good_cnt = len(predictions), 0
    p_cnt, r_cnt = 0, 0
    goodSingleCnt, goodMoreCnt = 0, 0
    singlePreCnt, singleRecallCnt = 0, 0
    morePreCnt, moreRecallCnt = 0, 0
    for i in range(all_cnt):
        pKeyphraseList, pSingleKpList, pMoreKpList = getKeyphraseList_top(predictions[i][0:len(groundtruth[i])], top_num)
        gKeyphraseList, gSingleKpList, gMoreKpList = getKeyphraseList(groundtruth[i])
        for p in pKeyphraseList:
            for g in gKeyphraseList:
                if p == g:
                    good_cnt += 1
                    break
        for p in pSingleKpList:
            for g in gSingleKpList:
                if p == g:
                    goodSingleCnt += 1
                    break
        for p in pMoreKpList:
            for g in gMoreKpList:
                if p == g:
                    goodMoreCnt += 1
                    break
        p_cnt += len(pKeyphraseList)  #######################
        r_cnt += len(gKeyphraseList)  #######################
        singlePreCnt += len(pSingleKpList)
        singleRecallCnt += len(gSingleKpList)
        morePreCnt += len(pMoreKpList)
        moreRecallCnt += len(gMoreKpList)
    res['a'] = 1.0*good_cnt/all_cnt
    res['p'] = 1.0*good_cnt/p_cnt
    res['r'] = 1.0*good_cnt/r_cnt
    res['f'] = 2.0*res['p']*res['r']/(res['p']+res['r'])
    res['SinglePre'] = 1.0 * goodSingleCnt/singlePreCnt
    res['SingleRecall'] = 1.0 * goodSingleCnt/singleRecallCnt
    res['SingleF1'] = 2.0 * res['SinglePre']*res['SingleRecall']/(res['SinglePre']+res['SingleRecall'])
    res['MorePre'] = 1.0 * goodMoreCnt / morePreCnt
    res['MoreRecall'] = 1.0 * goodMoreCnt / moreRecallCnt
    res['MoreF1'] = 2.0 * res['MorePre'] * res['MoreRecall'] / (res['MorePre'] + res['MoreRecall'])
    return res


