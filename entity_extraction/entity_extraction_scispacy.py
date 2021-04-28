import scispacy
import spacy
from scispacy.linking import EntityLinker
import time
import pandas as pd
import pickle

spacy.require_gpu()
nlp = spacy.load("en_core_sci_lg")
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
linker = nlp.get_pipe("scispacy_linker")

def text_entity_score(text):
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
    doc = nlp(text)
    return doc.ents




def main():
    
    df_train = pd.read_csv('/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/train_MEV.csv', encoding='utf-8')
    print (df_train.head(5))
    print ("DataFrame Size", df_train.shape)
    
    with open('/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/replacement_ent_score.pkl','wb') as f:
        pickle.dump(df_train['replacement'].apply(text_entity_score), f)
    
    with open('/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/replacement_google_ent_score.pkl','wb') as f:
        pickle.dump(df_train['replacement_google'].apply(text_entity_score), f)

    with open('/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/replacement_deepl_ent_score.pkl','wb') as f:
        pickle.dump(df_train['replacement_deepl'].apply(text_entity_score), f)

    with open('/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/replacement_ent.pkl','wb') as f:
        pickle.dump(df_train['replacement'].apply(extracts_entity), f)
    
    with open('/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/replacement_google_ent.pkl','wb') as f:
        pickle.dump(df_train['replacement_google'].apply(extracts_entity), f)

    with open('/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/replacement_deepl_ent.pkl','wb') as f:
        pickle.dump(df_train['replacement_deepl'].apply(extracts_entity), f)


if __name__ == "__main__":
    past_time = time.time()
    main()
    print ("Total_Time ({}(in Hrs) : {}(in Mins)".format((time.time()-past_time)/3600.0, (time.time()-past_time)/60.0))






