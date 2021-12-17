import json
import numpy as np
import random 
import pandas as pd

from sklearn.manifold import TSNE

gtLabelConsider = "entailment"
outlierFile = "MNLI/outlierFileCLSEmbeddings.json"
regularFile = "MNLI/regularFileCLSEmbeddings.json"

with open(outlierFile) as f:
    outlierData = json.load(f)

with open(regularFile) as f:
    regularData = json.load(f)

random.shuffle(regularData)
regularData = regularData[0:len(outlierData)]

concatData = []
for item in outlierData:
    #if item["gtLabel"] == gtLabelConsider:
    concatData.append(item["embedding"]+ [1])

for item in regularData:
    #if item["gtLabel"] == gtLabelConsider:
    concatData.append(item["embedding"]+ [0])

cols = [str(i) for i in range(len(outlierData[0]["embedding"]))] + ["class"]
df = pd.DataFrame(concatData, columns = cols)

tsne_em = TSNE(n_components=3, perplexity=30.0, n_iter=5000, verbose=1).fit_transform(df)
from bioinfokit.visuz import cluster
#cluster.tsneplot(score=tsne_em)

color_class = df['class'].to_numpy()
cluster.tsneplot(score=tsne_em, colorlist=color_class, dotsize = 3, valphadot=0.5, legendpos='upper right', legendanchor=(1.15, 1) )
