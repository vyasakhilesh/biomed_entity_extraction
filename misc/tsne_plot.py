from sklearn.manifold import TSNE
import plotly.express as px

df = pd.read_csv(path_output+'tsne_cn_emb_0_8.csv')
#print (df.head(5))
features = df.loc[:, 'V1':'V500']
print(features.shape)

tsne = TSNE(n_components=2, random_state=0)
projections = tsne.fit_transform(features)

fig = px.scatter(
    projections, x=0, y=1,
    color=df.Canonical_Name, labels={'color': 'Canonical_Name'}
)
#fig.show()
fig.write_html(path_output+"tsne_cn_emb_0_8.html")