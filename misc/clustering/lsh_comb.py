from LocalitySensitiveHashing import *
import pandas as pd
import time
import ast
import numpy as np

path = '/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/new/'
path_output = '/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/new/output/'

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
    return False 

def main():
    df_top_cn_CS_sem = pd.read_csv(path_output+'df_top_cn_CS_sem_all.csv', encoding='utf-8')
    print (df_top_cn_CS_sem.shape)
    #df_top_cn_CS_sem = df_top_cn_CS_sem.iloc[0:100]
    df_top_cn_CS_sem = df_top_cn_CS_sem.dropna(subset=['Canonical_Name']) 
    print (df_top_cn_CS_sem.shape)
    df_top_cn_CS_sem['isComb'] = df_top_cn_CS_sem['semantic_type'].apply(filter_cui_comb)
    df_top_cn_CS_sem = df_top_cn_CS_sem[df_top_cn_CS_sem['isComb']==True]
    print (df_top_cn_CS_sem.shape)


    df_pretrained_cui = pd.read_csv(path+'cui2vec_pretrained.csv')
    df_top_cui_list = df_top_cn_CS_sem.CUI.tolist()
    df_top_cui_cn_dict = dict(zip(df_top_cn_CS_sem.CUI, df_top_cn_CS_sem.Canonical_Name))
    print(len(df_top_cui_cn_dict))
    df_pretrained_cui_all = df_pretrained_cui[df_pretrained_cui['Unnamed: 0'].isin(df_top_cui_list)]
    print (df_pretrained_cui_all.shape)
    """df_pretrained_cui_all['Canonical_Name'] \
                        = df_pretrained_cui_all['Unnamed: 0'].apply(lambda cui : df_top_cui_cn_dict[cui])
    
    df_pretrained_cui_all['Canonical_Name'] \
        = df_pretrained_cui_all['Canonical_Name'].apply(lambda x: x.replace(',', ' ').replace(' ','_') )
    df_pretrained_cui_all.drop_duplicates(subset=['Canonical_Name'], inplace=True)
    columns_list = df_pretrained_cui_all.columns.tolist()
    columns = columns_list[-1:]+columns_list[1:-1]
    print (columns)
    df_pretrained_cui_all \
                    = df_pretrained_cui_all.loc[:,columns]"""
    
    # create sample groups
    expected_num_of_clusters = 150
    sample_list = ['sample'+str(i//expected_num_of_clusters)+'_'+str(i) \
                    for i in range(df_pretrained_cui_all.shape[0])]
    cui_sample_dict = dict(zip(df_pretrained_cui_all['Unnamed: 0'],sample_list))
    sample_cn_dict = {value:df_top_cui_cn_dict[key] for key, value in cui_sample_dict.items()}
    df_pretrained_cui_all['Unnamed: 0'] = df_pretrained_cui_all['Unnamed: 0'].apply(lambda x: cui_sample_dict[x])
    df_pretrained_cui_all.to_csv(path_output+'cui_data_lsh_comb_'+str(expected_num_of_clusters)+'.csv', index=False, header=False)
     
    datafile = path_output+'cui_data_lsh_comb_'+str(expected_num_of_clusters)+'.csv'
    lsh = LocalitySensitiveHashing(
                    datafile = datafile,
                    dim = 500,
                    r = 25,
                    b = 100,
                    expected_num_of_clusters = expected_num_of_clusters,
            )
    lsh.get_data_from_csv()
    lsh.initialize_hash_store()
    lsh.hash_all_data()
    similarity_groups = lsh.lsh_basic_for_neighborhood_clusters()
    #print (similarity_groups)
    coalesced_similarity_groups = lsh.merge_similarity_groups_with_coalescence( similarity_groups )
    #print (similarity_groups)
    merged_similarity_groups = lsh.merge_similarity_groups_with_l2norm_sample_based( coalesced_similarity_groups )
    #print (merged_similarity_groups)
    lsh.write_clusters_to_file( merged_similarity_groups, path_output+'clusters_comb_'+str(expected_num_of_clusters)+'.txt' )
    
    with open(path_output+'clusters_CN_comb_'+str(expected_num_of_clusters)+'.txt', 'w') as fw:
        with open(path_output+'clusters_comb_'+str(expected_num_of_clusters)+'.txt', 'r') as f:
            lines = []
            for line in f:
                line = list(ast.literal_eval(line))
                fw.writelines('{')
                fw.writelines(['"'+sample_cn_dict[i]+'", ' for i in line])
                fw.writelines('}')
                fw.writelines('\n\n')
                next(f)

if __name__ == "__main__":
    past_time = time.time()
    main()
    print ("Total_Time ({}(in Hrs) : {}(in Mins)".format((time.time()-past_time)/3600.0, (time.time()-past_time)/60.0))

