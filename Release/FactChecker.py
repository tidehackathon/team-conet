from sentence_transformers import SentenceTransformer, util
import nltk
import pandas as pd
import numpy as np

def cosine_similarity(model_name, source_sentence, sentences_to_check):
    model = SentenceTransformer(model_name)
    embeddings_source_sent = model.encode(source_sentence, convert_to_tensor=True)
    embeddings_sents_to_check = model.encode(sentences_to_check, convert_to_tensor=True)      
    cosine_scores = dict()
    for i in range(len(sentences_to_check)):
        cosine_score = util.cos_sim(embeddings_source_sent, embeddings_sents_to_check[i])
        # Position of sentence in the list is the key and score is the value 
        cosine_scores[i] = cosine_score.item()
    # Sort scores
    sort_cosine_scores = sorted(cosine_scores.items(), key=lambda x: x[1], reverse=True)
    return cosine_scores

# List of sentence simialrity models available in Hugging Face & with sentence transformers
sent_sim_model_names = [
    'sentence-transformers/all-MiniLM-L6-v2', 
    'sentence-transformers/all-mpnet-base-v2',
    'sentence-transformers/all-MiniLM-L12-v2',
    'sentence-transformers/all-distilroberta-v1',
    'sentence-transformers/all-roberta-large-v1',
    'sentence-transformers/all-mpnet-base-v2',
    'symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli',
    'flax-sentence-embeddings/all_datasets_v3_roberta-large',
    'sentence-transformers/sentence-t5-base',
    'sentence-transformers/all-MiniLM-L12-v1',
    'sentence-transformers/sentence-t5-xl',
    # 'sentence-transformers/sentence-t5-xxl', --> Huge for KI-Server
    'sentence-transformers/sentence-t5-large',
    'sentence-transformers/gtr-t5-base',
    'flax-sentence-embeddings/reddit_single-context_mpnet-base',
    'sentence-transformers/gtr-t5-xl',
    'sentence-transformers/all-MiniLM-L6-v1',
    #'sentence-transformers/gtr-t5-xxl',
    'flax-sentence-embeddings/all_datasets_v3_mpnet-base',
#    'digio/Twitter4SSE',
    'usc-isi/sbert-roberta-large-anli-mnli-snli',
#    'navteca/multi-qa-mpnet-base-cos-v1',
    'flax-sentence-embeddings/all_datasets_v4_mpnet-base',
    'flax-sentence-embeddings/all_datasets_v3_distilroberta-base',
    'flax-sentence-embeddings/all_datasets_v4_MiniLM-L6',
    'symanto/sn-mpnet-base-snli-mnli',
    'flax-sentence-embeddings/all_datasets_v3_MiniLM-L6',
    'flax-sentence-embeddings/all_datasets_v4_MiniLM-L12',
    'flax-sentence-embeddings/all_datasets_v3_MiniLM-L12',
    'arredondos/my_sentence_transformer',
    'multi-qa-distilbert-cos-v1',
    'multi-qa-MiniLM-L6-cos-v1',
    'paraphrase-multilingual-mpnet-base-v2',
    'paraphrase-albert-small-v2',
    'paraphrase-multilingual-MiniLM-L12-v2',
    'paraphrase-MiniLM-L3-v2',
    'distiluse-base-multilingual-cased-v1',
    'distiluse-base-multilingual-cased-v2',
    'distilbert-multilingual-nli-stsb-quora-ranking',
    'bert-base-nli-mean-tokens',
    'paraphrase-multilingual-mpnet-base-v2',
]
# References: 
# https://www.sbert.net/docs/pretrained_models.html 
# https://huggingface.co/models?language=en&library=sentence-transformers&sort=downloads

def test(inp, reliable, model_name):
    if "/" in model_name:
        model_name_p1, model_name_p2 = model_name.split("/")
    else:
        model_name_p1 = "others"
        model_name_p2 = model_name
    return(cosine_similarity(model_name, inp, reliable))

right=["Zelensky wants more weapons","Zelensky does not refuse more weapons"]
wrong=["Zelensky does not want more weapons","Zelensky refuses more weapons"]
reliable=["Zelensky wants to have more weapons","Zelensky does not refuse to receive more weapons"]
score=[0 for i in range(len(sent_sim_model_names))]
for i in range(len(sent_sim_model_names)):
    model_name=sent_sim_model_names[i]
    print(str(i+1), model_name)
    rightsum=[0 for j in range(len(reliable))]
    for j in range(len(right)):
        inp=right[j]
        erg=test(inp, reliable, model_name)
        print("  ",str(j+1), erg)
        rightsum = [rightsum[ii]+erg[ii] for ii in range(len(reliable))]
    rightsum=[ii/len(right) for ii in rightsum]
    wrongsum=[0 for j in range(len(reliable))]
    for j in range(len(wrong)):
        inp=wrong[j]
        erg=test(inp, reliable, model_name)
        print("  ",str(j+1), erg)
        wrongsum = [wrongsum[ii]+erg[ii] for ii in range(len(reliable))]
    wrongsum=[ii/len(wrong) for ii in wrongsum]
    rightsum=max(rightsum)
#    rightsumy=np.sqrt(1-rightsum*rightsum)
    wrongsum=min(wrongsum)
#    wrongsumy=np.sqrt(1-wrongsum*wrongsum)
    score[i]=rightsum-wrongsum
    print("quality: ", score[i])

print(score)
#[0.030603140592575073, 0.00951772928237915, 0.022241592407226562, 0.008863985538482666, 0.08568724989891052, 0.00951772928237915, 0.5151061788201332, 0.08568724989891052, 0.06282052397727966, 0.02475947141647339, 0.19554218649864197, 0.16602087020874023, 0.009562760591506958, 0.0032562315464019775, -0.0025464296340942383, 0.015270709991455078, 0.044637829065322876, 0.29048028588294983, 0.00951772928237915, 0.008863985538482666, 0.030603140592575073, 0.15297561883926392, 0.015270709991455078, 0.022241592407226562, 0.02475947141647339, 0.030603140592575073, -0.00891798734664917, 0.024734526872634888, 0.24280592799186707, 0.15598124265670776, 0.0663396418094635, 0.05877518653869629, 0.02411741018295288, 0.017913252115249634, 0.031981080770492554, 0.367337241768837, 0.24280592799186707]

besti=[i for i in range(len(score)) if score[i]==max(score)][0]
model_name=sent_sim_model_names[besti]
print(model_name)
#model_name="symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli"

with open("./EUDisinfo.txt") as f:
    fake = f.read().splitlines()

f.close()
fake=fake[0:-1]

# testing
##########
# prints the best and the worst fit of inp to the known disinformation in fake
inp='Russia takes back land that belongs to them'
res=cosine_similarity(model_name, inp, fake)
res={k: v for k, v in sorted(res.items(), key=lambda item: item[1],reverse=True)}
result=[i for i in res]
print("")
print(inp)
print(result)
print(fake[result[0]])
print(fake[result[-1]])

inp='The USA commit crimes'
res=cosine_similarity(model_name, inp, fake)
res={k: v for k, v in sorted(res.items(), key=lambda item: item[1],reverse=True)}
result=[i for i in res]
print("")
print(inp)
print(fake[result[0]])

inp='Western countries dont want to give weapons to Ukraine'
res=cosine_similarity(model_name, inp, fake)
res={k: v for k, v in sorted(res.items(), key=lambda item: item[1],reverse=True)}
result=[i for i in res]
print("")
print(inp)
print(fake[result[0]])

