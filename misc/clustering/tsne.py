import pandas as pd
import numpy as np
import pickle
import IPython.display as Displays
import time
from sklearn.manifold import TSNE
import plotly.express as px

pd.set_option('display.max_rows', 6000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 1000)

path = '/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/new/'
path_output = '/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/new/output/'

def tsne_cn_emb_0_8_plot():
    df = pd.read_csv(path_output+'tsne_cn_emb_0_8.csv')
    #print (df.head(5))
    features = df.loc[:, 'V1':'V500']
    print(features.shape)

    tsne = TSNE(n_components=2, random_state=0, perplexity=5, metric='cosine', square_distances=True)
    projections = tsne.fit_transform(features)

    fig = px.scatter(
        projections, x=0, y=1,
        color=df.Canonical_Name, labels={'color': 'Canonical_Name'}
    )
    #fig.show()
    fig.write_html(path_output+"tsne_cn_emb_0_8.html")

def tsne_cn_emb_plot():
    df = pd.read_csv(path_output+'tsne_cn_emb.csv')
    #print (df.head(5))
    features = df.loc[:, 'V1':'V500']
    print(features.shape)

    tsne = TSNE(n_components=2, random_state=0, perplexity=5, metric='cosine', square_distances=True)
    projections = tsne.fit_transform(features)

    fig = px.scatter(
        projections, x=0, y=1,
        color=df.Canonical_Name, labels={'color': 'Canonical_Name'}
    )
    #fig.show()
    fig.write_html(path_output+"tsne_cn_emb.html")



def main():
    df_top_cn_CS_sem = pd.read_csv(path_output+'df_top_cn_CS_sem_all.csv', encoding='utf-8')
    print(df_top_cn_CS_sem.shape)
    df_top_cn_CS_sem.dropna(inplace=True)
    print(df_top_cn_CS_sem.shape)
    df_top_cn_CS_sem.sort_values(by=['CUI_NN_CS_all'], ascending=False, inplace=True)
    
    # pretrained cui2vec
    df_pretrained_cui = pd.read_csv(path+'cui2vec_pretrained.csv')
    
    # for t-SNE
    t_SNE_cui_list = df_top_cn_CS_sem.CUI.tolist()+df_top_cn_CS_sem.CUI_NN_all.tolist()
    t_SNE_cui_CN_list = [str(10000+i)+'#'+j for i,j in enumerate(df_top_cn_CS_sem.Canonical_Name.tolist())]\
                        +[str(10000+i)+'##'+j for i,j in enumerate(df_top_cn_CS_sem.CUI_NN_CN_all.tolist())]
    print (len(t_SNE_cui_CN_list), len(t_SNE_cui_list))
    tsne_cui_cn_all_dict = dict(zip(t_SNE_cui_list, t_SNE_cui_CN_list))
    df_pretrained_cui_all = df_pretrained_cui[df_pretrained_cui['Unnamed: 0'].isin(t_SNE_cui_list)]
    print(df_pretrained_cui_all.shape)
    df_pretrained_cui_all['Canonical_Name'] = df_pretrained_cui_all['Unnamed: 0'].apply(lambda x : tsne_cui_cn_all_dict[x])
    df_pretrained_cui_all.sort_values(by=['Canonical_Name'],inplace=True)
    df_pretrained_cui_all.drop(columns=['Unnamed: 0', 'Canonical_Name'])\
                            .to_csv(path_output+'tsne_emb.csv', index=False, header=False)
    df_pretrained_cui_all[['Canonical_Name']].to_csv(path_output+'tsne_cn.csv', index=False, header=False)
    df_pretrained_cui_all.to_csv(path_output+'tsne_cn_emb.csv', index=False)
    # for t-SNE 0.8
    df_top_cn_CS_sem_0_8 = df_top_cn_CS_sem[df_top_cn_CS_sem['CUI_NN_CS_all']>=0.8]
    t_SNE_cui_list_0_8 = df_top_cn_CS_sem_0_8.CUI.tolist()\
                         +df_top_cn_CS_sem_0_8.CUI_NN_all.tolist()
    t_SNE_cui_CN_list_0_8 = [str(10000+i)+'#'+j for i,j in enumerate(df_top_cn_CS_sem_0_8.Canonical_Name.tolist())]\
                        +[str(10000+i)+'##'+j for i,j in enumerate(df_top_cn_CS_sem_0_8.CUI_NN_CN_all.tolist())]
    tsne_cui_cn_all_dict_0_8 = dict(zip(t_SNE_cui_list_0_8, t_SNE_cui_CN_list_0_8))
    print (len(t_SNE_cui_list_0_8), len(t_SNE_cui_CN_list_0_8))
    df_pretrained_cui_0_8 = df_pretrained_cui[df_pretrained_cui['Unnamed: 0'].isin(t_SNE_cui_list_0_8)]
    print(df_pretrained_cui_0_8.shape)
    df_pretrained_cui_0_8['Canonical_Name'] = df_pretrained_cui_0_8['Unnamed: 0'].apply(lambda x : tsne_cui_cn_all_dict_0_8[x])
    df_pretrained_cui_0_8.sort_values(by=['Canonical_Name'],inplace=True)
    df_pretrained_cui_0_8.drop(columns=['Unnamed: 0', 'Canonical_Name'])\
                    .to_csv(path_output+'tsne_emb_0_8.csv', index=False, header=False)
    df_pretrained_cui_0_8[['Canonical_Name']].to_csv(path_output+'tsne_cn_0_8.csv', index=False, header=False)
    df_pretrained_cui_0_8.to_csv(path_output+'tsne_cn_emb_0_8.csv', index=False)

if __name__ == "__main__":
    past_time = time.time()
    main()
    tsne_cn_emb_0_8_plot()
    tsne_cn_emb_plot()
    print ("Total_Time ({}(in Hrs) : {}(in Mins)".format((time.time()-past_time)/3600.0, (time.time()-past_time)/60.0))
