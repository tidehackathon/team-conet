from sentence_transformers import SentenceTransformer, util
import nltk
import pandas as pd

sent_sim_model_name = "paraphrase-multilingual-mpnet-base-v2"


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
"""    for i in sort_cosine_scores:
        for y in len(sentences_to_check):
            if i[0] == y:
                print(i, sentences_to_check(y))"""
    return cosine_scores

#news_df = pd.read_csv("/home/celine/Hackathon/Data/clean_articles.csv", engine='python',error_bad_lines=False)
#sentences_to_check = news_df["articles"]
cosine_similarity(sent_sim_model_name, "input satz", ["sentence to check for 1","sentence to check for 2"])
#cosine_similarity(sent_sim_model_name, "@EricLiptonNYT @MarkLandler @kbennhold @MatinaStevis 1.  Russia has lost Ukraine to western democracies for generations. 2.  Russia has collapsed.  Economy in chaos and the wolves are circling Putin. 3.  China will never invade Taiwan while Biden is president. 4.  Biden can now turn fully to China mano y mano. What a few weeks!", sentences_to_check)