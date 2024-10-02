from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import spacy
import re
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize FastAPI app
app = FastAPI()

# Load the spaCy NER model
model_path = "ner_model"  # Replace with your actual model path
nlp = spacy.load(model_path)

# Define a request body for text input
class TextRequest(BaseModel):
    text: str

# Text cleaning function
def clean_text(text: str) -> str:
    """
    Clean the input text by normalizing whitespaces, newlines, and other minor formatting issues.
    """
    # Remove multiple spaces and replace them with a single space
    return re.sub(r'\s+', ' ', text).strip()

# TF-IDF summarization function
def tfidf_summarize(text, top_n=3, remove_stopwords=True, min_sentence_len=5):
    """
    Summarize text by selecting the top N sentences with the highest TF-IDF scores.
    """
    if not text or len(text.strip()) == 0:
        return "Error: The input text is empty."

    # Clean and tokenize sentences
    sentences = sent_tokenize(text)
    cleaned_sentences = [clean_text(sentence) for sentence in sentences if len(sentence.split()) >= min_sentence_len]

    if len(cleaned_sentences) == 0:
        return "Error: No valid sentences found in the input text."

    # Apply TF-IDF vectorizer to the sentences
    vectorizer = TfidfVectorizer(stop_words='english' if remove_stopwords else None)
    sentence_vectors = vectorizer.fit_transform(cleaned_sentences)

    # Rank sentences by the sum of their TF-IDF scores
    norm_sentence_scores = np.array(sentence_vectors.sum(axis=1)).flatten()
    ranked_sentence_indices = norm_sentence_scores.argsort()[::-1]

    # Select the top N sentences for the summary
    ranked_sentences = [sentences[i] for i in ranked_sentence_indices[:top_n]]

    # Clean the output to ensure no line breaks or extra spaces
    summary = ' '.join(ranked_sentences)
    return summary

# Endpoint to process text and return named entities
@app.post("/predict")
async def predict(request: TextRequest):
    text = request.text

    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    # Clean the input text
    cleaned_text = clean_text(text)

    # Process the text with the spaCy model
    doc = nlp(cleaned_text)

    entities = []

    # Check if we have an ABSTRACT entity and apply TF-IDF summarization
    for ent in doc.ents:
        entity_info = {
            "text": ent.text,
            "label": ent.label_.upper(),
            "start": ent.start_char,
            "end": ent.end_char
        }

        # If the entity is an ABSTRACT, replace the 'text' with the summarized text
        if ent.label_.upper() == "ABSTRACT":
            summarized_abstract = tfidf_summarize(ent.text, top_n=6, remove_stopwords=True, min_sentence_len=6)  
            entity_info["text"] = summarized_abstract  # Replace the text with the summarized version
        
        entities.append(entity_info)

    return {"entities": entities}

# Health check route
@app.get("/")
async def health_check():
    return {"status": "Model is running"}
