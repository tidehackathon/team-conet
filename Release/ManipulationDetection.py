#https://simpletransformers.ai/docs/classification-models/
#https://github.com/ThilinaRajapakse/simpletransformers
#https://simpletransformers.ai/docs/binary-classification/
import pandas as pd 
import pickle
import torch
from simpletransformers.classification import ClassificationModel, ClassificationArgs

def isNaN(num):
    return num != num

def getcsv(path):
    rows = pd.read_csv(path, chunksize=5000000) 
    i, chuck=next(x for x in enumerate(rows))
    chuck.to_csv('out{}.csv'.format(i))
    return([i for i in chuck['body'] if not isNaN(i)])

with open("./annotated.tweets", "rb") as fp:
    annotated=pickle.load(fp)

fp.close()
print(len(annotated))
print("Example of disinformation:\n",annotated[0],"\n\n")
print("Example of information:\n",annotated[-1],"\n\n")

#model = ClassificationModel(
#    "roberta",
#    "outputs_20230223",
#)

#docs=getcsv("./DisInformation-Challenge-Data/russian_invasion_of_ukraine.csv")
#docs=docs[0:min(10000,len(docs))]
#print(len(docs),"\n\n")
#predictions, raw_outputs = model.predict(docs)
#raw=list(raw_outputs[:,1])
#result=[[docs[i],raw[i]] for i in range(len(docs))]
#res = sorted(result, key=lambda x: (x[1], x[0]))
#print("Detected Information:",res[0],"\n\n")
#print("Detected Disinformation:",res[-1],"\n\n")
#print("Detected Disinformation:",res[-2],"\n\n")
#print("Detected Disinformation:",res[-3],"\n\n")


train_data=annotated
train_df = pd.DataFrame(train_data)
train_df.columns = ["text", "labels"]

model_args = ClassificationArgs()
model_args.num_train_epochs = 10
model_args.overwrite_output_dir = True
model_args.use_cuda=torch.cuda.is_available()
model = ClassificationModel("roberta", "roberta-base", args=model_args)
model.train_model(train_df)

docs=getcsv("./DisInformation-Challenge-Data/russian_invasion_of_ukraine.csv")
docs=docs[0:min(10000,len(docs))]
print(len(docs),"\n\n")
predictions, raw_outputs = model.predict(docs)
raw=list(raw_outputs[:,1])
result=[[docs[i],raw[i]] for i in range(len(docs))]
res = sorted(result, key=lambda x: (x[1], x[0]))
print("Detected Information:",res[0],"\n\n")
print("Detected Disinformation:",res[-1],"\n\n")
print("Detected Disinformation:",res[-2],"\n\n")
print("Detected Disinformation:",res[-3],"\n\n")




