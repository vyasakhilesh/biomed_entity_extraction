import scispacy
import spacy
from scispacy.linking import EntityLinker
import time
import pandas as pd
import pickle

spacy.require_gpu()
nlp = spacy.load("en_core_sci_lg")
nlp.add_pipe("scispacy_linker", config={"k":30,"resolve_abbreviations": True, "linker_name": "umls", 
                "filter_for_definitions":False, "threshold":0.7})
linker = nlp.get_pipe("scispacy_linker")

def text_entity_score(text):
    text = str(text)
    doc = nlp(text)
    # print(list(doc.sents))
    # print(doc.ents)

    # print ('First Entity')
    # print (doc.ents[0])
    # entity = doc.ents[1]
    #for umls_ent in entity._.kb_ents:
    #    print(linker.kb.cui_to_entity[umls_ent[0]])

    # return [linker.kb.cui_to_entity[umls_ent] for ent in doc.ents for umls_ent in ent._.kb_ents]
    umls_l = []
    for ent in doc.ents:
        umls_l.append([umls_ent for umls_ent in ent._.kb_ents])
    
    # for ent in doc.ents:
    #     for umls_ent in ent._.kb_ents:
    #         print(umls_ent, end=',')
    #     print('\n')
    return umls_l

def extracts_entity(text):
    text = str(text)
    doc = nlp(text)
    return doc.ents

def extracts_entity_len(text):
    text = str(text)
    doc = nlp(text)
    return len(doc.ents)




def main():
    path = '/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/new/'
    path_output = '/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/new/output/'
    df_train = pd.read_csv(path+'train_MEV.csv', encoding='utf-8')
    print (df_train.head(5))
    print ("DataFrame Size", df_train.shape)
    
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

    df_train['ents_len'] = df_train['replacement'].apply(extracts_entity_len)
    df_train['google_ents_len'] = df_train['google'].apply(extracts_entity_len)
    df_train['deepl_ents_len'] = df_train['deepl'].apply(extracts_entity_len)
    df_train['google_abbr_ents_len'] = df_train['google_abbr'].apply(extracts_entity_len)
    df_train['deepl_abbr_ents_len'] = df_train['deepl_abbr'].apply(extracts_entity_len)


    df_train = df_train.to_csv(path+'train_MEV_ents.csv', encoding='utf-8', index=False)


if __name__ == "__main__":
    past_time = time.time()
    main()
    print ("Total_Time ({}(in Hrs) : {}(in Mins)".format((time.time()-past_time)/3600.0, (time.time()-past_time)/60.0))






