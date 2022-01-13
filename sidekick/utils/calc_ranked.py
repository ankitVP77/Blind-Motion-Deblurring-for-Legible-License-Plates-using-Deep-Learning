import numpy as np

def calc_ranked(preds, labels):

    rank1=0
    rank5=0

    for p,l in zip(preds, labels):

        p= np.argsort(p)[::-1]

        if l in p[:5]:
            rank5+=1

        if l==p[0]:
            rank1+=1

    rank1= float(rank1/len(preds))
    rank5= float(rank5/len(preds))

    return rank1, rank5