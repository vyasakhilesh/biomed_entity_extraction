import numpy as np
import ast

filter_comb = ['Sign or Symptom', 'Disease or Syndrome', 'Congenital Abnormality', 
               'Anatomical Abnormality', 'Pathologic Function', 'Acquired Abnormality', 
               'Mental or Behavioral Dysfunction']



def filter_cui_comb(semantic_type_label_string):
    try:
        semantic_type_label_list = ast.literal_eval(semantic_type_label_string)
        for sem_type in semantic_type_label_list:
            if sem_type in filter_comb:
                return True
    except:
        return np.nan
    """semantic_type_label_list = ast.literal_eval(semantic_type_label_string)
    for sem_type in semantic_type_label_list:
            if sem_type in filter_comb:
                return True"""
    return False 


def main():
    pass

if __name__ == '__main__':
    main()