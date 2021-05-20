import scispacy
import spacy
from scispacy.linking import EntityLinker
from scispacy.abbreviation import AbbreviationDetector
import time
import pandas as pd
import pickle
import numpy as np
import IPython.display as Displays
pd.options.display.width = 0
pd.set_option('display.max_rows', 6000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 1000)


spacy.require_gpu()
nlp = spacy.load("en_core_sci_lg")
nlp.add_pipe("scispacy_linker", config={"k":30,"resolve_abbreviations": True, "linker_name": "umls", 
                "filter_for_definitions":False, "threshold":0.7})
linker = nlp.get_pipe("scispacy_linker")

def text_entity_cui_flat(text, top=3):
    if text != text:
        text = ''
    umls_l = []
    try:
        doc = nlp(text)
        for ent in doc.ents:
            umls_l = umls_l+[umls_ent[0] for i, umls_ent in enumerate(ent._.kb_ents) if umls_ent[1]==1.0 or i<top]
    except:
        pass
    return list(set(umls_l))

def extracts_entity(text):
    if text != text:
        text = ''
    doc = nlp(str(text))
    return doc.ents

def extracts_entity_len(text):
    if text != text:
        text = ''
    doc = nlp(str(text))
    return len(doc.ents)



def main():
    path = '/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/new/'
    path_output = '/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/new/output/'
    
    df = pd.read_csv(path_output+'spanish_google_deep_last.csv', encoding='utf-8')\
        .loc[:,['Value_ID','value','replacement','google_abbr','deepl_abbr']]
    Displays.display(df.head(5))
    Displays.display ("DataFrame Size", df.shape)
    
    df['mev_cui'] = df['replacement'].apply(text_entity_cui_flat)
    df['google_abbr_cui'] =  df['google_abbr'].apply(text_entity_cui_flat)
    df['deepl_abbr_cui'] =  df['deepl_abbr'].apply(text_entity_cui_flat)
    
    Displays.display(df.head(10))

    df['mev_cui_len']=df['mev_cui'].apply(lambda x: len(x))
    df['google_deepl_abbr_cui']=(df['google_abbr_cui']+df['deepl_abbr_cui']).apply(lambda x: list(set(x)))
    # df['ID'] = ['ID'+ str(x) for x in df.index.tolist()]
    df['TEXT_ID'] = df['Value_ID']
    df['TEXT'] = df['value']
    df['FINAL_CUI'] = np.where(df['mev_cui_len']!=0, df['mev_cui'], df['google_deepl_abbr_cui'])
    df['FINAL_CUI_len'] = df['FINAL_CUI'].apply(lambda x:len(x))
    df_id_cui = df[['TEXT_ID','FINAL_CUI']]
    df_id_cui_expand = df_id_cui.explode('FINAL_CUI', ignore_index=True)
    
    Displays.display(df.head(10))
    Displays.display(df_id_cui_expand.head(10))
    # save to files
    df.to_csv(path_output+'id_text_cuiLst.csv', encoding='utf-8', index=False)
    df_id_cui_expand.drop_duplicates().to_csv(path_output+'id_cuis_kg.csv', encoding='utf-8', index=False)
    df[['TEXT_ID','TEXT']].drop_duplicates().to_csv(path_output+'id_text_kg.csv', encoding='utf-8', index=False)
    

if __name__ == "__main__":
    past_time = time.time()
    main()
    print ("Total_Time ({}(in Hrs) : {}(in Mins)".format((time.time()-past_time)/3600.0, (time.time()-past_time)/60.0))






