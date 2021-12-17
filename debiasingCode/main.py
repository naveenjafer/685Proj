import json
import os
import csv
import sys

csv.field_size_limit(sys.maxsize)

# hyperparams
M = 3
T = 2

data = []
with open(os.path.join("MNLI", "train.tsv")) as f:
    read_tsv = csv.reader(f, delimiter="\t")
    for row in read_tsv:
        data.append([row[-3], row[-1]])

data = data[1:]

patternMap = {}

print(data[0])
for item in data[0:10]:
    hyp = item[0]
    label = item[1]
    hypList = hyp.split(" ")
    for index,token in enumerate(hypList):
        prefix = token
        for m in range(1,M+1):
            for t in range(1,T+1):
                print("M: ", m, " T:", t)
                def recurse(prefix, t_left, m_left, word_ind):
                    print(prefix)
                    if word_ind >= len(hypList)-1:
                        return
                    if m_left == 0:
                        if prefix not in patternMap:
                            patternMap[prefix] = {"count" : 0, "neutral" : 0, "entailment" : 0, "contradiction" : 0}
                        patternMap[prefix]["count"] += 1
                        patternMap[prefix][label] += 1
                        return
                    if t_left == 0:
                        recurse(prefix + " " + hypList[word_ind+1], T, m_left-1, word_ind+1)
                        return
                    else:
                        recurse(prefix + " " + hypList[word_ind+1], T, m_left-1, word_ind+1)
                        recurse(prefix + " _$_", t_left-1, m_left, word_ind+1)
                recurse(prefix, t, m-1, index)

print(patternMap)

with open("pickle.json", "w") as f:
    json.dump(patternMap, f)
                    
print(data[0:2])




