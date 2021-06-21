import time
import pandas as pd
import pickle
import numpy as np
import requests

headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
url = 'https://labs.tib.eu/sdm/biofalcon/api?mode=long'

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

def extracts_entity_len_series(series):
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
    path_output = '/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/new/output_biofalcon_slow/'
    df_train = pd.read_csv(path_output+'train_MEV.csv', encoding='utf-8')
    print (df_train.head(5))
    print ("DataFrame Size", df_train.shape)
    
    with open(path_output+'value_ent_score.pkl','wb') as f:
        pickle.dump(text_entity_score_series(df_train['value']), f)
    
    with open(path_output+'value_abbr_ent_score.pkl','wb') as f:
        pickle.dump(text_entity_score_series(df_train['value_witht_abbr']), f)
    
    with open(path_output+'ent_score.pkl','wb') as f:
        pickle.dump(text_entity_score_series(df_train['replacement']), f)
    
    with open(path_output+'google_ent_score.pkl','wb') as f:
        pickle.dump(text_entity_score_series(df_train['google']), f)

    with open(path_output+'deepl_ent_score.pkl','wb') as f:
        pickle.dump(text_entity_score_series(df_train['deepl']), f)

     
    with open(path_output+'google_abbr_ent_score.pkl','wb') as f:
        pickle.dump(text_entity_score_series(df_train['google_abbr']), f)

    with open(path_output+'deepl_abbr_ent_score.pkl','wb') as f:
        pickle.dump(text_entity_score_series(df_train['deepl_abbr']), f)
    

    # Without Stop_words
    
    with open(path_output+'google_withst_num_ent_score.pkl','wb') as f:
        pickle.dump(text_entity_score_series(df_train['google_withst_num']), f)

    with open(path_output+'deepl_withst_num_ent_score.pkl','wb') as f:
        pickle.dump(text_entity_score_series(df_train['deepl_withst_num']), f)

    with open(path_output+'google_abbr_withst_num_ent_score.pkl','wb') as f:
        pickle.dump(text_entity_score_series(df_train['google_abbr_withst_num']), f)

    with open(path_output+'deepl_abbr_withst_num_ent_score.pkl','wb') as f:
        pickle.dump(text_entity_score_series(df_train['deepl_abbr_withst_num']), f)
    
    # extract entites
    df_train['value_ents'] = extracts_entity_series(df_train['value'])
    df_train['value_abbr_ents'] = extracts_entity_series(df_train['value_witht_abbr'])
    df_train['ents'] = extracts_entity_series(df_train['replacement'])
    df_train['google_ents'] = extracts_entity_series(df_train['google'])
    df_train['deepl_ents'] = extracts_entity_series(df_train['deepl'])

    df_train['google_abbr_ents'] = extracts_entity_series(df_train['google_abbr'])
    df_train['deepl_abbr_ents'] = extracts_entity_series(df_train['deepl_abbr'])
    
    # extract entites len
    df_train['value_ents_len'] = extracts_entity_len_series(df_train['value'])
    df_train['value_abbr_ents_len'] = extracts_entity_len_series(df_train['value_witht_abbr'])
    df_train['ents_len'] = extracts_entity_len_series(df_train['replacement'])
    df_train['google_ents_len'] = extracts_entity_len_series(df_train['google'])
    df_train['deepl_ents_len'] = extracts_entity_len_series(df_train['deepl'])
    df_train['google_abbr_ents_len'] = extracts_entity_len_series(df_train['google_abbr'])
    df_train['deepl_abbr_ents_len'] = extracts_entity_len_series(df_train['deepl_abbr'])


    df_train = df_train.to_csv(path_output+'train_MEV_ents.csv', encoding='utf-8', index=False)


if __name__ == "__main__":
    past_time = time.time()
    main()
    print ("Total_Time ({}(in Hrs) : {}(in Mins)\n\n\n".format((time.time()-past_time)/3600.0, (time.time()-past_time)/60.0))






