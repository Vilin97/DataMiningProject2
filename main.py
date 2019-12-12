import pandas as pd
import numpy as np

# !!!

# change the path to where train_data.csv, test_data.csv and random_solution.csv are.

# !!!
path = ""

predict_data = pd.read_csv(path+"random_solution.csv")
predict_df = pd.DataFrame(predict_data)

train_data = pd.read_csv(path+"train_data.csv")
train_df = pd.DataFrame(train_data)

test_data = pd.read_csv(path+"test_data.csv")
test_df = pd.DataFrame(test_data)

def process_ref(references):
    # creates an array of strings for each entry in train_data
    references = [s[:-1] for s in references.split()] # makes an array of references from the string
    if ['['] == references:
        references = []
    else:
        references[0] = references[0][1:]
    return references

train_df['references'] = train_df['references'].apply(process_ref)

# create two dictionaries:
# bigdic has form (paper,[year, venue, [references]])
# targets has form (paper, [papers that cite it])
targets = dict()
bigdic = dict()
for i in range(len(train_df)):
    paper_id, year, venue, references = train_df.iloc[i]
    bigdic[paper_id] = [year, venue, references]
    for paper in references:
        if paper in targets:
            targets[paper].append(paper_id)
        else:
            targets[paper] = [paper_id]

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def normalized_inner_product(a,b):
    if len(a) == 0 or len(b) == 0:
        return 0
    return len(a.intersection(b))/min(len(a),len(b))


def predict0(source, target):
    inadj = 0
    same_journal = 0
    refadj = 0
    if target in targets:
        inadj = ((len(targets[target]) - 30)/3000)*0.25
    if source in bigdic and target in bigdic:
        year1,venue1,ref1 = bigdic[source]
        year2,venue2,ref2 = bigdic[target]
        if year1 < year2:
            return 0
        same_journal = (int(venue1 == venue2) - 0.07)*0.25
        refadj = (normalized_inner_product(set(ref1),set(ref2)) - 0.02)*0.5
    res = 0.5 + inadj + same_journal + refadj
    res = min(res,1)
    res = max(0,res)
    return res

# make the prediction and save in predict_df
prediction_vector = np.zeros(len(predict_df))
for i in range(len(test_df)):
    source,target,edge_id = test_df.iloc[i]
    prediction_vector[edge_id] = predict0(source,target)
predict_df["edge_present"] = prediction_vector

# Save output
predict_df.to_csv(path+r'output.csv',index=False)
