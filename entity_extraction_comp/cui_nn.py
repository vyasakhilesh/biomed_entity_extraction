import pandas as pd
import numpy as np
import pickle
import spacy
import IPython.display as Displays
import scispacy
from scispacy.linking import EntityLinker
import time

pd.options.display.width = 0
pd.set_option('display.max_rows', 6000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 1000)


spacy.require_gpu()
nlp = spacy.load("en_core_sci_lg")
nlp.add_pipe("scispacy_linker", config={"k":30,"resolve_abbreviations": True, 
               "linker_name": "umls", 
                "filter_for_definitions":True, "threshold":0.7})
linker = nlp.get_pipe("scispacy_linker")


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

#paths
path = '/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/new/'
path_output = '/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/new/output/'

# load cui embeddings
with open(path+'embeddings_cui.pkl', 'rb') as f:    
    embeddings_cui = pickle.load(f)
    
embeddings_cui_imp = {key:value for key, value in embeddings_cui.items() if key in imp_cui_dict}
embeddings_cui = dict(list(embeddings_cui.items())[0:10])

def get_canonical_name(cui_id):
    try:
        return linker.kb.cui_to_entity[cui_id].canonical_name
    except:
        return ''

def get_cosine(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def nearest_cui_cs_imp(cui):
    if cui in embeddings_cui:
        cui_emb = embeddings_cui[cui]
        prev_score = -1.0
        pre_cui = ''
        for cuii, emb in embeddings_cui_imp.items():
            score = get_cosine(cui_emb, emb)
            if score > prev_score:
                pre_cui = cuii
                prev_score = score
        return (pre_cui, prev_score)
    else:
        return ('Others',np.nan)

def nearest_cui_cs_all(cui):
    if cui in embeddings_cui_imp:
        return (cui, 1.0)
    if cui in embeddings_cui:
        embeddings_cui_tmp = embeddings_cui.copy()
        del embeddings_cui_tmp[cui]
        cui_emb = embeddings_cui[cui]
        prev_score = -1.0
        pre_cui = ''
        for cuii, emb in embeddings_cui_tmp.items():
            score = get_cosine(cui_emb, emb)
            if score > prev_score:
                pre_cui = cuii
                prev_score = score
        return (pre_cui, prev_score)
    else:
        return ('Others',np.nan)

def main():
    df_id_cui_cn = pd.read_csv(path_output+'id_cuis_cn.csv', encoding='utf-8')
    
    df_count_cn = df_id_cui_cn.groupby(['FINAL_CUI'])[['TEXT_ID']].count()
    df_top_cn = df_count_cn.sort_values(['TEXT_ID'],ascending=False)
    cui_cn_dict = dict(zip(df_id_cui_cn.FINAL_CUI, df_id_cui_cn.Canonical_Name))

    df_top_cn['CUI'] = df_top_cn.index
    df_top_cn['Canonical_Name'] = df_top_cn['CUI'].apply(lambda x : cui_cn_dict[x])
    df_top_cn['TEXT_ID_COUNT'] = df_top_cn['TEXT_ID']

    #imp
    cui_nn_cs_imp = df_top_cn['CUI'].apply(lambda x : nearest_cui_cs_imp(x))
    df_top_cn['CUI_NN_imp'] = [i for i,_ in cui_nn_cs_imp]
    df_top_cn['CUI_NN_CN_imp'] = df_top_cn['CUI_NN_imp'].apply(lambda x : cui_cn_dict[x] if x in cui_cn_dict else '')
    df_top_cn['CUI_NN_CS_imp'] = [j for _,j in cui_nn_cs_imp]

    # All
    cui_nn_cs_all = df_top_cn['CUI'].apply(lambda x : nearest_cui_cs_all(x))
    df_top_cn['CUI_NN_all'] = [i for i,_ in cui_nn_cs_all]
    df_top_cn['CUI_NN_CN_all'] = df_top_cn['CUI_NN_all'].apply(lambda x : get_canonical_name(x))
    df_top_cn['CUI_NN_CS_all'] = [j for _,j in cui_nn_cs_all]
    
    df_top_cn.sort_values(by=['CUI_NN_CS_all'], inplace=True, ascending=False)
    Displays.display(df_top_cn.head(10))



if __name__ == "__main__":
    past_time = time.time()
    main()
    print ("Total_Time ({}(in Hrs) : {}(in Mins)".format((time.time()-past_time)/3600.0, (time.time()-past_time)/60.0))

