import time
import pandas as pd
import pickle
import numpy as np
import requests

headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
url = 'https://labs.tib.eu/sdm/biofalcon/api?mode=long'




def text_entity_score(text, top=1):
    if text != text:
        text = ''
    payload = '{"text":"'+text+'"}'
    r = requests.post(url, data=payload.encode('utf-8'), headers=headers)
    if r.status_code == 200:
        response=r.json()
        #print(response)
        if len(response['entities_UMLS'])!=0:
            #print (text, response['entities_UMLS'])
            return [[(men_cui[1],men_cui[0]) for men_cui in response['entities_UMLS']]]
        else:
            return [[]]
    else:
        return [[]]


def extracts_entity(text):
    if text != text:
        text = ''
    payload = '{"text":"'+text+'"}'
    r = requests.post(url, data=payload.encode('utf-8'), headers=headers)
    if r.status_code == 200:
        response=r.json()
        #print(response)
        if len(response['entities_UMLS'])!=0:
            #print (text, response['entities_UMLS'])
            return [men_cui[0] for men_cui in response['entities_UMLS']]
        else:
            return []
    else:
        return []

def extracts_entity_len(text):
    if text != text:
        text = ''
    payload = '{"text":"'+text+'"}'
    r = requests.post(url, data=payload.encode('utf-8'), headers=headers)
    if r.status_code == 200:
        response=r.json()
        #print(response)
        if len(response['entities_UMLS'])!=0:
            # print (text, response['entities_UMLS'])
            return len([men_cui[0] for men_cui in response['entities_UMLS']])
        else:
            return len([])
    else:
        return len([])


def text_entity_score_series(series, top=1):
    ss = []
    for index, text in series.items():
        if text != text:
            text = ''
        payload = '{"text":"'+text+'"}'
        r = requests.post(url, data=payload.encode('utf-8'), headers=headers)
        if r.status_code == 200:
            response=r.json()
            #print(response)
            if len(response['entities_UMLS'])!=0:
                #print (text, response['entities_UMLS'])
                ss.append([[(men_cui[1],men_cui[0]) for men_cui in response['entities_UMLS']]])
            else:
                ss.append([[]])
        else:
            ss.append([[]])
    return ss


def extracts_entity_series(series):
    ss = []
    for index, text in series.items():
        if text != text:
            text = ''
        payload = '{"text":"'+text+'"}'
        r = requests.post(url, data=payload.encode('utf-8'), headers=headers)
        if r.status_code == 200:
            response=r.json()
            #print(response)
            if len(response['entities_UMLS'])!=0:
                #print (text, response['entities_UMLS'])
                ss.append([men_cui[0] for men_cui in response['entities_UMLS']])
            else:
                ss.append([])
        else:
            ss.append([])
    return ss

def extracts_entity_series(series):
    ss = []
    for index, text in series.items():
        if text != text:
            text = ''
        payload = '{"text":"'+text+'"}'
        r = requests.post(url, data=payload.encode('utf-8'), headers=headers)
        if r.status_code == 200:
            response=r.json()
            #print(response)
            if len(response['entities_UMLS'])!=0:
                # print (text, response['entities_UMLS'])
                ss.append(len([men_cui[0] for men_cui in response['entities_UMLS']]))
            else:
                ss.append(len([]))
        else:
            ss.append(len([]))
    return ss

def main():
    path = '/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/new/'
    path_output = '/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/new/output_biofalcon/'
    df_train = pd.read_csv(path_output+'train_MEV.csv', encoding='utf-8')
    print (df_train.head(5))
    print ("DataFrame Size", df_train.shape)
    
    with open(path_output+'value_ent_score.pkl','wb') as f:
        pickle.dump(df_train['value'].apply(text_entity_score), f)
    
    with open(path_output+'value_abbr_ent_score.pkl','wb') as f:
        pickle.dump(df_train['value_witht_abbr'].apply(text_entity_score), f)
    
    with open(path_output+'ent_score.pkl','wb') as f:
        pickle.dump(df_train['replacement'].apply(text_entity_score), f)
    
    with open(path_output+'google_ent_score.pkl','wb') as f:
        pickle.dump(df_train['google'].apply(text_entity_score), f)

    with open(path_output+'deepl_ent_score.pkl','wb') as f:
        pickle.dump(df_train['deepl'].apply(text_entity_score), f)

     
    with open(path_output+'google_abbr_ent_score.pkl','wb') as f:
        pickle.dump(df_train['google_abbr'].apply(text_entity_score), f)

    with open(path_output+'deepl_abbr_ent_score.pkl','wb') as f:
        pickle.dump(df_train['deepl_abbr'].apply(text_entity_score), f)
    

    # Without Stop_words
    
    with open(path_output+'google_withst_num_ent_score.pkl','wb') as f:
        pickle.dump(df_train['google_withst_num'].apply(text_entity_score), f)

    with open(path_output+'deepl_withst_num_ent_score.pkl','wb') as f:
        pickle.dump(df_train['deepl_withst_num'].apply(text_entity_score), f)

    with open(path_output+'google_abbr_withst_num_ent_score.pkl','wb') as f:
        pickle.dump(df_train['google_abbr_withst_num'].apply(text_entity_score), f)

    with open(path_output+'deepl_abbr_withst_num_ent_score.pkl','wb') as f:
        pickle.dump(df_train['deepl_abbr_withst_num'].apply(text_entity_score), f)

    df_train['ents'] = df_train['replacement'].apply(extracts_entity)
    df_train['google_ents'] = df_train['google'].apply(extracts_entity)
    df_train['deepl_ents'] = df_train['deepl'].apply(extracts_entity)

    df_train['google_abbr_ents'] = df_train['google_abbr'].apply(extracts_entity)
    df_train['deepl_abbr_ents'] = df_train['deepl_abbr'].apply(extracts_entity)
    
    df_train['value_ents_len'] = df_train['value'].apply(extracts_entity_len)
    df_train['value_abbr_ents_len'] = df_train['value_abbr'].apply(extracts_entity_len)
    df_train['ents_len'] = df_train['replacement'].apply(extracts_entity_len)
    df_train['google_ents_len'] = df_train['google'].apply(extracts_entity_len)
    df_train['deepl_ents_len'] = df_train['deepl'].apply(extracts_entity_len)
    df_train['google_abbr_ents_len'] = df_train['google_abbr'].apply(extracts_entity_len)
    df_train['deepl_abbr_ents_len'] = df_train['deepl_abbr'].apply(extracts_entity_len)


    df_train = df_train.to_csv(path_output+'train_MEV_ents.csv', encoding='utf-8', index=False)


if __name__ == "__main__":
    past_time = time.time()
    main()
    print ("Total_Time ({}(in Hrs) : {}(in Mins)\n\n\n".format((time.time()-past_time)/3600.0, (time.time()-past_time)/60.0))






