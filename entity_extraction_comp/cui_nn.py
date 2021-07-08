import pandas as pd
import numpy as np
import pickle
import spacy
import IPython.display as Displays
import scispacy
from scispacy.linking import EntityLinker
import time
from operator import itemgetter
import ast

pd.options.display.width = 0
pd.set_option('display.max_rows', 6000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 1000)


# spacy.require_gpu()
# nlp = spacy.load("en_core_sci_lg")
# nlp.add_pipe("scispacy_linker", config={"k":30,"resolve_abbreviations": True, 
#                "linker_name": "umls", 
#                 "filter_for_definitions":True, "threshold":0.7})
# linker = nlp.get_pipe("scispacy_linker")


# Important CUIs
imp_cui_dict = {'C0004096':'Asthma','C0018799':'Cardiopathy', 
'C0011849':'Diabetes Mellitus (DM)' , 'C0242339':'Dyslipidemia',
'C0024117':'Chronic obstructive pulmonary disease' ,
'C0001973':'Alcoholism/Ex Alcoholism','C0019158':'Hepatitis',
'C0020443':'Hypercholesterolemia' ,'C0020538':'HTA','C0022658':'Renal disease',
'C0028754':'Obesity','C0003467':'Depressive syndrome / Anxiety',
'C0041296':'Tuberculosis','C0042373':'Vascular disease',
'Others':'Others','C0023895':'Liver disease','C1559265':'Gastrointestinal',
'C0524851':'Neurodegenerative disorder',
'C1704272':'Benign prostatic Hyperplasia','C0520679':'Obstructive sleep Apnea'}

filter_comb = ['Sign or Symptom', 'Disease or Syndrome', 'Congenital Abnormality', 
               'Anatomical Abnormality', 'Pathologic Function', 'Acquired Abnormality', 
               'Mental or Behavioral Dysfunction']

#paths
path = '/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/new/'
path_output = '/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/new/output/'


#sementic type filtered
# df_sem_type_com = pd.read_csv(path+'semantic_type_comb.csv', encoding='ISO-8859-1')
# cui_comb_dict = dict(zip(df_sem_type_com['CUI'], df_sem_type_com['String']))
# cui_comb_list = df_sem_type_com['CUI'].tolist()

def filter_cui_comb(semantic_type_label_string):
    semantic_type_label_list = ast.literal_eval(semantic_type_label_string)
    for sem_type in semantic_type_label_list:
        if sem_type in filter_comb:
            return True
        return False 

# Canonical_Names
df_sem_type_tib = pd.read_csv(path+'UMLS_combined.csv',encoding='ISO-8859-1')
cui_CN_dict = dict(zip(df_sem_type_tib['CUI'], df_sem_type_tib['Label']))
cui_SEMType_dict = dict(zip(df_sem_type_tib['CUI'], df_sem_type_tib['Semantic_type_label']))
#for comb
cui_comb_list = df_sem_type_tib

#for all


# load cui embeddings
with open(path+'embeddings_cui.pkl', 'rb') as f:    
    embeddings_cui = pickle.load(f)


embeddings_cui_imp = {key:value for key, value in embeddings_cui.items() if key in imp_cui_dict}

embeddings_cui_comb = {cui:cui_emb for cui, cui_emb in embeddings_cui.items() if cui in cui_comb_list}
embeddings_cui_imp_comb = {key:value for key, value in embeddings_cui.items() if key in imp_cui_dict 
                      and key in cui_comb_list}
#embeddings_cui = dict(list(embeddings_cui.items())[0:10])



def get_canonical_name(cui_id):
    try:
        # return linker.kb.cui_to_entity[cui_id].canonical_name
        return cui_CN_dict[cui_id]
    except:
        return ''

def get_cosine(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def nearest_cui_cs_imp_comb(cui):
    if cui in embeddings_cui_imp_comb:
        return (cui, 1.0)
    if cui in embeddings_cui_comb:
        cui_emb = embeddings_cui_comb[cui]
        return max([(cuii,get_cosine(cui_emb, emb)) for cuii, emb in embeddings_cui_imp_comb.items()],key=itemgetter(1))
    else:
        return ('',np.nan)

def nearest_cui_cs_all_comb(cui):
    if cui in embeddings_cui_imp_comb:
        return (cui, 1.0)
    if cui in embeddings_cui_comb:
        embeddings_cui_tmp_comb = embeddings_cui_comb.copy()
        del embeddings_cui_tmp_comb[cui]
        cui_emb = embeddings_cui_comb[cui]
        return max([(cuii,get_cosine(cui_emb, emb)) for cuii, emb in embeddings_cui_tmp_comb.items()],key=itemgetter(1))
        
    else:
        return ('',np.nan)

def nearest_cui_cs_imp(cui):
    if cui in embeddings_cui_imp:
        return (cui, 1.0)
    if cui in embeddings_cui:
        cui_emb = embeddings_cui[cui]
        return max([(cuii,get_cosine(cui_emb, emb)) for cuii, emb in embeddings_cui_imp.items()],key=itemgetter(1))
    else:
        return ('',np.nan)

def have_same_sem_type(cui1, cui2):
    if cui1 in cui_SEMType_dict and cui2 in cui_SEMType_dict:
        if len(set(cui_SEMType_dict[cui1]).intersection(set(cui_SEMType_dict[cui2])))>0:
            return True
    return False

def nearest_cui_cs_all(cui):
    if cui in embeddings_cui_imp:
        return (cui, 1.0)
    if cui in embeddings_cui:
        embeddings_cui_tmp = embeddings_cui.copy()
        del embeddings_cui_tmp[cui]
        cui_emb = embeddings_cui[cui]
        return max([(cuii,get_cosine(cui_emb, emb)) for cuii, emb in embeddings_cui_tmp.items()\
                    if have_same_sem_type(cui,cuii)],key=itemgetter(1), default=('',np.nan))
        
    else:
        return ('',np.nan)


def extract_semantic_type(cui):
    if cui in cui_SEMType_dict:
        return cui_SEMType_dict[cui]
    return ''

def main():
    df_id_cui_cn = pd.read_csv(path_output+'id_cuis_cn.csv', encoding='utf-8')
    df_count_cn = df_id_cui_cn.groupby(['FINAL_CUI'])[['TEXT_ID']].count()
    df_top_cn = df_count_cn.sort_values(['TEXT_ID'],ascending=False)
    cui_cn_dict = dict(zip(df_id_cui_cn.FINAL_CUI, df_id_cui_cn.Canonical_Name))

    df_top_cn['CUI'] = df_top_cn.index.tolist()
    df_top_cn['Canonical_Name'] = df_top_cn['CUI'].apply(lambda x : cui_cn_dict[x])
    df_top_cn.rename(columns={'TEXT_ID':'TEXT_ID_COUNT'},inplace=True)
    df_top_cn = df_top_cn[['CUI','Canonical_Name','TEXT_ID_COUNT']]
    df_top_cn.reset_index(drop=True, inplace=True)
    print (df_top_cn.head(5))
    

    #df_top_cn = df_top_cn.iloc[0:100]
    # Add semantic type 
    df_top_cn['semantic_type'] = df_top_cn['CUI'].apply(lambda cui : extract_semantic_type(cui))
    
    
    # #imp
    # cui_nn_cs_imp = df_top_cn['CUI'].apply(lambda x : nearest_cui_cs_imp(x))
    # df_top_cn['CUI_NN_imp'] = [i for i,_ in cui_nn_cs_imp]
    # df_top_cn['CUI_NN_CN_imp'] = df_top_cn['CUI_NN_imp'].apply(lambda x : cui_cn_dict[x] if x in cui_cn_dict else '')
    # df_top_cn['CUI_NN_CS_imp'] = [j for _,j in cui_nn_cs_imp]

    # # All
    # cui_nn_cs_all = df_top_cn['CUI'].apply(lambda x : nearest_cui_cs_all(x))
    # df_top_cn['CUI_NN_all'] = [i for i,_ in cui_nn_cs_all]
    # df_top_cn['CUI_NN_CN_all'] = df_top_cn['CUI_NN_all'].apply(lambda x : get_canonical_name(x))
    # df_top_cn['CUI_NN_CS_all'] = [j for _,j in cui_nn_cs_all]
    
    cui_nn_cs_all = df_top_cn['CUI'].apply(lambda x : nearest_cui_cs_all(x))
    df_top_cn['CUI_NN_all'] = [i for i,_ in cui_nn_cs_all]
    df_top_cn['CUI_NN_CN_all'] = df_top_cn['CUI_NN_all'].apply(lambda x : get_canonical_name(x))
    df_top_cn['CUI_NN_CS_all'] = [j for _,j in cui_nn_cs_all]

    # df_top_cn.sort_values(by=['CUI_NN_CS_all'], inplace=True, ascending=False)
    df_top_cn.to_csv(path_output+'df_top_cn_CS_sem_all.csv', encoding='utf-8', index=False)

    Displays.display(df_top_cn.head(10))



if __name__ == "__main__":
    past_time = time.time()
    main()
    print ("Total_Time ({}(in Hrs) : {}(in Mins)".format((time.time()-past_time)/3600.0, (time.time()-past_time)/60.0))

