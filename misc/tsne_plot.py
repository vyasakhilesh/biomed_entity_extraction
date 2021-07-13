from sklearn.manifold import TSNE
import plotly.express as px
import pandas as pd
import time

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
    tsne_cn_emb_0_8_plot()
    tsne_cn_emb_plot()

if __name__ == "__main__":
    past_time = time.time()
    main()
    print ("Total_Time ({}(in Hrs) : {}(in Mins)".format((time.time()-past_time)/3600.0, (time.time()-past_time)/60.0))


