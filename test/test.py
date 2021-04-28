import scispacy
import spacy
from scispacy.linking import EntityLinker
import time

past_time = time.time()
spacy.require_gpu()

nlp = spacy.load("en_core_sci_lg")
text = """
Myeloid derived suppressor cells (MDSC) are immature 
myeloid cells with immunosuppressive activity. 
They accumulate in tumor-bearing mice and humans 
with different types of cancer, including hepatocellular 
carcinoma (HCC).
"""
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})

doc = nlp(text)

print(list(doc.sents))
# >>> ["Myeloid derived suppressor cells (MDSC) are immature myeloid cells with immunosuppressive activity.", 
#     "They accumulate in tumor-bearing mice and humans with different types of cancer, including hepatocellular carcinoma (HCC)."]

# Examine the entities extracted by the mention detector.
# Note that they don't have types like in SpaCy, and they
# are more general (e.g including verbs) - these are any
# spans which might be an entity in UMLS, a large
# biomedical database.
print(doc.ents)
# >>> (Myeloid derived suppressor cells,
#      MDSC,
#      immature,
#      myeloid cells,
#      immunosuppressive activity,
#      accumulate,
#      tumor-bearing mice,
#      humans,
#      cancer,
#      hepatocellular carcinoma,
#      HCC)

print ('Second Entity')
print (doc.ents[1])

entity = doc.ents[1]

linker = nlp.get_pipe("scispacy_linker")

for umls_ent in entity._.kb_ents:
	print(linker.kb.cui_to_entity[umls_ent[0]])

# We can also visualise dependency parses
# (This renders automatically inside a jupyter notebook!):

#from spacy import displacy
#displacy.render(next(doc.sents), style='dep', jupyter=True)

# See below for the generated SVG.
# Zoom your browser in a bit!

print ("Total_Time ({}(in Hrs) : {}(in Mins)".format((time.time()-past_time)/3600.0, (time.time()-past_time)/60.0))

