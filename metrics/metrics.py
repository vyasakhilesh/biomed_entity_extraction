# define jaccard distance
import pickle
import pandas as pd
import time

def jaccard_similarity(l1, l2):
    s1 = set(l1)
    s2 = set(l2)
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    try:
        return intersection / float(union)
    except:
        return 1.0

def jaccard_arr(arr1, arr2):
    score_l = []
    for t1_l, t2_l in zip(arr1, arr2):
        print (t1_l)
        print (t2_l)
        j_l = []
        for e1_s_l, e2_s_l in zip(t1_l, t2_l):
            e1_l = [e1 for e1, _ in e1_s_l]
            e2_l = [e2 for e2, _ in e2_s_l]
            j_l.append(jaccard_similarity(e1_l, e2_l))
        print (j_l)
        try:
            score = sum(j_l)/float(len(j_l))
        except:
            score = 0.0
        score_l.append(score)
    return score_l

def jaccard_arr_flat(arr1, arr2, top=5):
    score_l = []
    for t1_l, t2_l in zip(arr1, arr2):
        e1_l = [cui for i in t1_l for cui, _ in i]
        e2_l = [cui for i in t2_l for cui, _ in i]
        score = jaccard_similarity(e1_l, e2_l)
        print (t1_l)
        print (e1_l)
        print (t2_l)
        print (e2_l)
        print(score)
        score_l.append(score)
    return score_l



def main():
    df_train = pd.read_csv('/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/train_MEV.csv', encoding='utf-8')
    print (df_train.head(5))
    print ("DataFrame Size", df_train.shape)
    
    with open('/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/replacement_ent_score.pkl','rb') as f:
        mev_arr = pickle.load(f)
    
    with open('/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/replacement_google_ent_score.pkl','rb') as f:
        google_arr = pickle.load(f)

    with open('/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/replacement_deepl_ent_score.pkl','rb') as f:
        deepl_arr = pickle.load(f)
    
    # jaccard_arr(mev_arr, google_arr)
    # jaccard_arr(mev_arr, deepl_arr)
    df_train['Avg_JD_MEV_google'] = jaccard_arr_flat(mev_arr, google_arr)
    df_train['Avg_JD_MEV_deepl'] = jaccard_arr_flat(mev_arr, deepl_arr)

    print (df_train.head(5))
    print (df_train.tail(5))

if __name__ == "__main__":
    past_time = time.time()
    main()
    print ("Total_Time ({}(in Hrs) : {}(in Mins)".format((time.time()-past_time)/3600.0, (time.time()-past_time)/60.0))










