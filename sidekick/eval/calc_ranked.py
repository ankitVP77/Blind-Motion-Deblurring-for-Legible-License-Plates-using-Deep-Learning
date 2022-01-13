import numpy as np

def calculate_ranked(preds, labels):
    rank1=0
    rank5=0

    for p,l in zip(preds, labels):
        #sort preds in descending order of their confidence and return the indices of these
        p= np.argsort(p)[::-1]

        # checking for rank5
        if l in p[:5]:
            rank5+=1
        # checking rank1
        if l==p[0]:
            rank1+=1


    # Final accuracies
    rank1= rank1/len(labels)
    rank5= rank5/len(labels)

    return rank1,rank5