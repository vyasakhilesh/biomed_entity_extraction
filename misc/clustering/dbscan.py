from sklearn import cluster
import time
from sklearn.manifold import TSNE
import pandas as pd
import plotly.express as px
import numpy as np
import ast

path = '/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/new/'
path_output = '/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/new/output/'

def dbscan_algo(data_array, eps = 225, min_samples = 20):
    dbscan = cluster.DBSCAN(eps=eps, min_samples=min_samples, metric='cosine',)
    clustering_labels = dbscan.fit_predict(data_array)
    return clustering_labels

def tsne_algo(data_array):
    X_embedded = TSNE(n_components=2).fit_transform(data_array)
    return X_embedded

def plot(df, eps, min_samples):
    fig = px.scatter(df, x="x_component", y="y_component", hover_name="Canonical_Name", color = "labels", size_max=60)
    fig.update_layout(height=800)
    fig.write_html(path_output+'dbscan_tsne'+'_'+str(eps)+'_'+str(min_samples)+'.html')

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
    df_top_cn_CS_sem = pd.read_csv(path_output+'df_top_cn_CS_sem_all.csv', encoding='utf-8')
    print(df_top_cn_CS_sem.shape)
    df_top_cn_CS_sem = df_top_cn_CS_sem.dropna(subset=['Canonical_Name']) 
    print (df_top_cn_CS_sem.shape)
    df_top_cn_CS_sem['isComb'] = df_top_cn_CS_sem['semantic_type'].apply(filter_cui_comb)
    df_top_cn_CS_sem = df_top_cn_CS_sem[df_top_cn_CS_sem['isComb']==True]
    print (df_top_cn_CS_sem.shape)


    # for t-SNE
    t_SNE_cui_list = df_top_cn_CS_sem.CUI.tolist()
    t_SNE_cui_CN_list =df_top_cn_CS_sem.Canonical_Name.tolist() #[str(10000+i)+'#'+j for i,j in enumerate(df_top_cn_CS_sem.Canonical_Name.tolist())]
    print (len(t_SNE_cui_CN_list), len(t_SNE_cui_list))
    tsne_cui_cn_all_dict = dict(zip(t_SNE_cui_list, t_SNE_cui_CN_list))

    # pretrained cui2vec
    df_pretrained_cui = pd.read_csv(path+'cui2vec_pretrained.csv')
    df_pretrained_cui_all = df_pretrained_cui[df_pretrained_cui['Unnamed: 0'].isin(t_SNE_cui_list)]
    print(df_pretrained_cui_all.shape)
    df_pretrained_cui_all['Canonical_Name'] = df_pretrained_cui_all['Unnamed: 0'].apply(lambda x : tsne_cui_cn_all_dict[x])
    df_pretrained_cui_all.sort_values(by=['Canonical_Name'],inplace=True)
    #df_pretrained_cui_all.drop(columns=['Unnamed: 0', 'Canonical_Name'])\
    #                        .to_csv(path_output+'tsne_emb_org_comb_5.csv', index=False, header=False)
    #df_pretrained_cui_all[['Canonical_Name']].to_csv(path_output+'tsne_cn_org_comb_5.csv', index=False, header=False)
    #df_pretrained_cui_all.iloc[:,1:].to_csv(path_output+'tsne_cn_emb_org_comb_5.csv', index=False)
    

    data_array = df_pretrained_cui_all.drop(columns=['Unnamed: 0', 'Canonical_Name']).to_numpy()
    # dbscan
    eps = 0.1
    min_samples = 2
    df_pretrained_cui_all['labels'] = dbscan_algo(data_array, eps, min_samples)

    # tsne
    X_embedded = tsne_algo(data_array)
    df_pretrained_cui_all["x_component"]=X_embedded[:,0]
    df_pretrained_cui_all["y_component"]=X_embedded[:,1]

    # plot
    plot(df_pretrained_cui_all, eps, min_samples)

    df_pretrained_cui_all[['Canonical_Name','labels']].groupby(by=['labels'])['Canonical_Name']\
        .apply(lambda x: x.tolist()).reset_index()\
        .to_csv(path_output+'dbscan_tsne_comb'+'_'+str(eps)+'_'+str(min_samples)+'.csv', index=False)

if __name__ == "__main__":
    past_time = time.time()
    main()
    print ("Total_Time ({}(in Hrs) : {}(in Mins)".format((time.time()-past_time)/3600.0, (time.time()-past_time)/60.0))

