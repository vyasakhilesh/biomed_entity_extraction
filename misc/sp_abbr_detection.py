import spacy
from scispacy.abbreviation import AbbreviationDetector
import pandas as pd

path='/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/new/'
path_output = '/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/new/output/'
    
df = pd.read_csv(path+'comorbidities_modified_witht_abbr_new.csv', sep=',', error_bad_lines=True, encoding='utf-8')
print (df.shape)


nlp = spacy.load("es_dep_news_trf") # spanish

# Add the abbreviation pipe to the spacy pipeline.
nlp.add_pipe("abbreviation_detector")

doc = nlp("")
print("\t\t Abbreviation  (start, end)", "\t", "Definition")
for value in df['value']:
    doc = nlp(value)
    if len(doc._.abbreviations) > 0:
        print ("Text:", value)
        for abrv in doc._.abbreviations:
            print(f"\t\t{abrv} \t ({abrv.start}, {abrv.end}) {abrv._.long_form}")
