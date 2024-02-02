from flask import Flask,request,render_template
import sentence_transformers
from sentence_transformers import SentenceTransformer, util
import gensim
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import simphile
from simphile import jaccard_similarity,euclidian_similarity,compression_similarity


app=Flask(__name__)

# Load pre-trained Sentence Transformers model
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

# Function to calculate similarity between two text pairs
def calculate_similarity(text1, text2):
    # Encode the input texts
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    # Compute similarity score using dot product
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return similarity_score

def average_word_vector(tokens, model, dimension=100):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return np.zeros(dimension)
    return np.mean(vectors, axis=0)

def calculate_cosine_similarity(text1,text2):
    tokenized_text1 = text1.lower().split()
    tokenized_text2 = text2.lower().split()
    model = Word2Vec([tokenized_text1, tokenized_text2], vector_size=100, window=5, min_count=1, workers=4)
    # Calculate average word vectors for each text
    vector1 = average_word_vector(tokenized_text1, model)
    vector2 = average_word_vector(tokenized_text2, model)
    # Calculate cosine similarity
    similarity_score = cosine_similarity([vector1], [vector2])[0][0] 
    return similarity_score

# nuclear body seeks new tech
#terror suspects face arrest

@app.route("/",methods=['POST','GET'])
def text_smiliarity():
    if request.method=='POST':
        # Post method
        # getting text data from Form
        text1=request.form["text1"]
        text2=request.form["text2"]
        option=request.form["dropdown"]
        #Based on options calling out different models
        if option=="Hugging Faces":
            models_pred=calculate_similarity(text1, text2)
        elif option=="jaccard_similarity":
            models_pred=jaccard_similarity(text1, text2)
        elif option=="euclidean_similarity":
            models_pred=euclidian_similarity(text1, text2)
        elif option=="compression_similarity":
            models_pred=compression_similarity(text1, text2)
        elif option=="Cosine_similarity":
            models_pred=calculate_cosine_similarity(text1,text2)
        
        if models_pred==0:
           models_pred= "0.0"
        # passing data to HTML TEMPLATE
        return render_template("index.html",model_pred=models_pred)
    return render_template("index.html",model_pred="")
if __name__=="__main__":
    app.run(debug=True)