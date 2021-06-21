# define jaccard distance
import pickle
import pandas as pd
import time
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def jaccard_similarity(l1, l2):
    s1 = set(l1)
    s2 = set(l2)
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    try:
        return intersection / float(union)
    except:
        return np.nan

def jaccard_similarity_A(l1, l2):
    s1 = set(l1)
    s2 = set(l2)
    intersection = len(s1.intersection(s2))
    union = len(s1)
    try:
        return intersection / float(union)
    except:
        return np.nan

def overlap_coefficient(l1, l2):
    s1 = set(l1)
    s2 = set(l2)
    intersection = len(s1.intersection(s2))
    min_c = min(len(s1), len(s2))
    try:
        return intersection / min_c
    except:
        return np.nan

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

def jaccard_arr_flat(arr1, arr2):
    score_l = []
    for t1_l, t2_l in zip(arr1, arr2):
        e1_l = [cui for i in t1_l for cui, _ in i]
        e2_l = [cui for i in t2_l for cui, _ in i]
        score = jaccard_similarity(e1_l, e2_l)
        score_l.append(score)
    return score_l

def oc_arr_flat(arr1, arr2):
    score_l = []
    for t1_l, t2_l in zip(arr1, arr2):
        e1_l = [cui for i in t1_l for cui, _ in i]
        e2_l = [cui for i in t2_l for cui, _ in i]
        score = overlap_coefficient(e1_l, e2_l)
        score_l.append(score)
    return score_l


def jaccard_arr_flat_A(arr1, arr2):
    score_l = []
    for t1_l, t2_l in zip(arr1, arr2):
        e1_l = [cui for i in t1_l for cui, _ in i]
        e2_l = [cui for i in t2_l for cui, _ in i]
        score = jaccard_similarity_A(e1_l, e2_l)
        score_l.append(score)
    return score_l

def jaccard_arr_flat_com(arr1, arr2, arr3):
    score_l = []
    for t1_l, t2_l, t3_l in zip(arr1, arr2, arr3):
        e1_l = [cui for i in t1_l for cui, _ in i]
        e2_l = [cui for i in t2_l for cui, _ in i]
        e3_l = [cui for i in t3_l for cui, _ in i]
        e4_l = list(set(e2_l).intersection(set(e3_l)))
        score = jaccard_similarity(e1_l, e4_l)
        score_l.append(score)
    return score_l

def jaccard_arr_flat_com_A(arr1, arr2, arr3):
    score_l = []
    for t1_l, t2_l, t3_l in zip(arr1, arr2, arr3):
        e1_l = [cui for i in t1_l for cui, _ in i]
        e2_l = [cui for i in t2_l for cui, _ in i]
        e3_l = [cui for i in t3_l for cui, _ in i]
        e4_l = list(set(e2_l).intersection(set(e3_l)))
        score = jaccard_similarity_A(e1_l, e4_l)
        score_l.append(score)
    return score_l

def jaccard_arr_flat_un(arr1, arr2, arr3):
    score_l = []
    for t1_l, t2_l, t3_l in zip(arr1, arr2, arr3):
        e1_l = [cui for i in t1_l for cui, _ in i]
        e2_l = [cui for i in t2_l for cui, _ in i]
        e3_l = [cui for i in t3_l for cui, _ in i]
        e4_l = list(set(e2_l).union(set(e3_l)))
        score = jaccard_similarity(e1_l, e4_l)
        score_l.append(score)
    return score_l

def jaccard_arr_flat_un_A(arr1, arr2, arr3):
    score_l = []
    for t1_l, t2_l, t3_l in zip(arr1, arr2, arr3):
        e1_l = [cui for i in t1_l for cui, _ in i]
        e2_l = [cui for i in t2_l for cui, _ in i]
        e3_l = [cui for i in t3_l for cui, _ in i]
        e4_l = list(set(e2_l).union(set(e3_l)))
        score = jaccard_similarity_A(e1_l, e4_l)
        score_l.append(score)
    return score_l



"""def precision_arr_flat(arr1, arr2, top=5):
    score_l = []
    for t1_l, t2_l in zip(arr1, arr2):
        e1_l = [cui for i in t1_l for cui, _ in i]
        e2_l = [cui for i in t2_l for cui, _ in i]
        score = precision_score(e1_l, e2_l, average='macro')
        score_l.append(score)
    return score_l

def recall_arr_flat(arr1, arr2, top=5):
    score_l = []
    for t1_l, t2_l in zip(arr1, arr2):
        e1_l = [cui for i in t1_l for cui, _ in i]
        e2_l = [cui for i in t2_l for cui, _ in i]
        score = recall_score(e1_l, e2_l, average='macro')
        score_l.append(score)
    return score_l """



def main():

    path = '/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/new/'
    path_output = '/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/new/output_0.4.0/'

    df_train = pd.read_csv(path_output+'train_MEV_ents.csv', encoding='utf-8')
    print (df_train.head(5))
    print ("DataFrame Size", df_train.shape)
    
    with open(path_output+'ent_score.pkl','rb') as f:
        mev_arr = pickle.load(f)
    
    with open(path_output+'google_ent_score.pkl','rb') as f:
        google_arr = pickle.load(f)

    with open(path_output+'deepl_ent_score.pkl','rb') as f:
        deepl_arr = pickle.load(f)
    
    with open(path_output+'google_withst_num_ent_score.pkl','rb') as f:
        google_withst_num_arr = pickle.load(f)

    with open(path_output+'deepl_withst_num_ent_score.pkl','rb') as f:
        deepl_withst_num_arr = pickle.load(f)

    with open(path_output+'google_abbr_ent_score.pkl','rb') as f:
        google_abbr_arr = pickle.load(f)

    with open(path_output+'deepl_abbr_ent_score.pkl','rb') as f:
        deepl_abbr_arr = pickle.load(f)
    
    with open(path_output+'google_abbr_withst_num_ent_score.pkl','rb') as f:
        google_abbr_withst_num_arr = pickle.load(f)

    with open(path_output+'deepl_abbr_withst_num_ent_score.pkl','rb') as f:
        deepl_abbr_withst_num_arr = pickle.load(f)


    # jaccard_arr(mev_arr, google_arr)
    # jaccard_arr(mev_arr, deepl_arr)

    df_train['mev_cui_score'] = mev_arr
    df_train['google_cui_score'] = google_arr
    df_train['deepl_cui_score'] = deepl_arr
    df_train['google_abbr_cui_score'] = google_abbr_arr
    df_train['deepl_abbr_cui_score'] = deepl_abbr_arr


    df_train['mev_cui_len'] = df_train['mev_cui_score'].apply(lambda x: sum([len(element) for element in x]))
    df_train['google_cui_len'] = df_train['google_cui_score'].apply(lambda x: sum([len(element) for element in x]))
    df_train['deepl_cui_len'] = df_train['deepl_cui_score'].apply(lambda x: sum([len(element) for element in x]))
    df_train['google_abbr_cui_len'] = df_train['google_abbr_cui_score'].apply(lambda x: sum([len(element) for element in x]))
    df_train['deepl_abbr_cui_len'] = df_train['deepl_abbr_cui_score'].apply(lambda x: sum([len(element) for element in x]))


    df_train['cui_jacD_mev_google'] = jaccard_arr_flat(mev_arr, google_arr)
    df_train['cui_jacD_mev_deepl'] = jaccard_arr_flat(mev_arr, deepl_arr)

    df_train['cui_jacD_mev_google_withst_num'] = jaccard_arr_flat(mev_arr, google_withst_num_arr)
    df_train['cui_jacD_mev_deepl_withst_num'] = jaccard_arr_flat(mev_arr, deepl_withst_num_arr)

    df_train['cui_jacD_mev_google_A'] = jaccard_arr_flat_A(mev_arr, google_arr)
    df_train['cui_jacD_mev_deepl_A'] = jaccard_arr_flat_A(mev_arr, deepl_arr)

    df_train['cui_jacD_mev_google_withst_num_A'] = jaccard_arr_flat_A(mev_arr, google_withst_num_arr)
    df_train['cui_jacD_mev_deepl_withst_num_A'] = jaccard_arr_flat_A(mev_arr, deepl_withst_num_arr)

    df_train['cui_jacD_mev_google_abbr'] = jaccard_arr_flat(mev_arr, google_abbr_arr)
    df_train['cui_jacD_mev_deepl_abbr'] = jaccard_arr_flat(mev_arr, deepl_abbr_arr)

    df_train['cui_jacD_mev_google_abbr_withst_num'] = jaccard_arr_flat(mev_arr, google_abbr_withst_num_arr)
    df_train['cui_jacD_mev_deepl_abbr_withst_num'] = jaccard_arr_flat(mev_arr, deepl_abbr_withst_num_arr)

    df_train['cui_jacD_mev_google_abbr_A'] = jaccard_arr_flat_A(mev_arr, google_abbr_arr)
    df_train['cui_jacD_mev_deepl_abbr_A'] = jaccard_arr_flat_A(mev_arr, deepl_abbr_arr)

    df_train['cui_jacD_mev_google_abbr_withst_num_A'] = jaccard_arr_flat_A(mev_arr, google_abbr_withst_num_arr)
    df_train['cui_jacD_mev_deepl_abbr_withst_num_A'] = jaccard_arr_flat_A(mev_arr, deepl_abbr_withst_num_arr)
    
    # Only Top Common
    df_train['cui_jacD_mev_google_deepl_com'] = jaccard_arr_flat_com(mev_arr, google_arr, deepl_arr)
    df_train['cui_jacD_mev_google_deepl_com_A'] = jaccard_arr_flat_com_A(mev_arr, google_arr, deepl_arr)
    df_train['cui_jacD_mev_google_deepl_abbr_com'] = jaccard_arr_flat_com(mev_arr, google_abbr_arr, deepl_abbr_arr)
    df_train['cui_jacD_mev_google_deepl_abbr_com_A'] = jaccard_arr_flat_com_A(mev_arr, google_abbr_arr, deepl_abbr_arr)

    # Only Top Union
    df_train['cui_jacD_mev_google_deepl_un'] = jaccard_arr_flat_un(mev_arr, google_arr, deepl_arr)
    df_train['cui_jacD_mev_google_deepl_un_A'] = jaccard_arr_flat_un_A(mev_arr, google_arr, deepl_arr)
    df_train['cui_jacD_mev_google_deepl_abbr_un'] = jaccard_arr_flat_un(mev_arr, google_abbr_arr, deepl_abbr_arr)
    df_train['cui_jacD_mev_google_deepl_abbr_un_A'] = jaccard_arr_flat_un_A(mev_arr, google_abbr_arr, deepl_abbr_arr)


    # overlap coefficient
    df_train['cui_OC_mev_google'] = oc_arr_flat(mev_arr, google_arr)
    df_train['cui_OC_mev_deepl'] = oc_arr_flat(mev_arr, deepl_arr)
    df_train['cui_OC_mev_google_abbr'] = oc_arr_flat(mev_arr, google_abbr_arr)
    df_train['cui_OC_mev_deepl_abbr'] = oc_arr_flat(mev_arr, deepl_abbr_arr)

    """df_train['macro_pre_cui_mev_google'] = precision_arr_flat(mev_arr, google_arr)
    df_train['macro_pre_cui_mev_deepl'] = precision_arr_flat(mev_arr, deepl_arr)
    df_train['macro_rec_cui_mev_google'] = recall_arr_flat(mev_arr, google_arr)
    df_train['macro_rec_cui_mev_deepl'] = recall_arr_flat(mev_arr, deepl_arr)"""


    print (df_train.head(5))
    print (df_train.tail(5))

    df_train = df_train.to_csv(path_output+'train_MEV_ents_jaccard.csv', encoding='utf-8', index=False)

if __name__ == "__main__":
    past_time = time.time()
    main()
    print ("Total_Time ({}(in Hrs) : {}(in Mins)".format((time.time()-past_time)/3600.0, (time.time()-past_time)/60.0))










