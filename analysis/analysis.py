import json
import numpy as np
import random 

datasetName = "QNLI"
outlierFile = datasetName + "/outlierFileCLSEmbeddings.json"
regularFile = datasetName + "/regularFileCLSEmbeddings.json"

gtLabelConsider = "entailment"

with open(outlierFile) as f:
    outlierData = json.load(f)

with open(regularFile) as f:
    regularData = json.load(f)

random.shuffle(regularData)
regularData = regularData[0:len(outlierData)]
#subsample regular data

for index,item in enumerate(outlierData):
    if True:
        outlierData[index]["class"] = 1
        outlierData[index]["embedding"] = np.array( outlierData[index]["embedding"] )

for index,item in enumerate(regularData):
    if True:
        regularData[index]["class"] = 0
        regularData[index]["embedding"] = np.array( regularData[index]["embedding"] )
    #print(regularData[index]["embedding"])
    
K = [1,2,5,10]
#np.linalg.norm()
allStats = []

print("Calculating for k")
stats_k = {}
dist_stats_k = {}
for k in K:
    stats_k[k] = {"1" : 0, "0" : 0}
    dist_stats_k[k] = {"1" : 0, "0" : 0}

stats = {"1": 0, "0" : 0}
for index, item in enumerate(outlierData):
    if "class" not in item:
        continue
    if index%100 == 0:
        print("Processing: ", index)
    distLists = []
    for otherItems in outlierData:
        if "class" not in otherItems:
            continue
        distLists.append([np.linalg.norm(item["embedding"]-otherItems["embedding"]), otherItems["class"]])
    
    for otherItems in regularData:
        if "class" not in otherItems:
            continue
        distLists.append([np.linalg.norm(item["embedding"]-otherItems["embedding"]), otherItems["class"]])
    distLists.sort(key=lambda x: x[0])
    for k in K:
        distListsTemp = distLists[1:k+1]
        #print(distListsTemp)
        counts = {"1": 0, "0" : 0}
        for distanceItem in distListsTemp:
            counts[str(distanceItem[1])] += 1
            stats_k[k][str(distanceItem[1])] += 1
            dist_stats_k[k][str(distanceItem[1])] += distanceItem[0]

for k in K:
    for key in stats_k[k]:
        stats_k[k][key] = round(stats_k[k][key]/(k*len(outlierData)),4)
        dist_stats_k[k][key] = round(dist_stats_k[k][key]/stats_k[k][key],8)

for index, k in enumerate(K):
    print("For k = ", k)
    print(stats_k[k])
    print("Distance stats")
    print(dist_stats_k[k])
    

    


