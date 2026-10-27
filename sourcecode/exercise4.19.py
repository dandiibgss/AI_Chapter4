# Exercise 4.19 - Sentiment analysis dengan beberapa kalimat positif & negatif
# pip install transformers torch
import os
os.environ["USE_TF"] = "0"
from transformers import pipeline
classifier = pipeline('sentiment-analysis')

sentences = [
    "This is a good movie.",                       # positif
    "I absolutely love this phone, the camera is amazing!",   # positif
    "The service was excellent and the staff were very friendly.",  # positif
    "This is the worst movie I have ever seen.",   # negatif
    "The food was cold and the waiter was rude.",  # negatif
    "I am so disappointed, the product broke after one day.",  # negatif
    "The movie was not bad.",                      # tricky: negasi
    "It is a table.",                              # netral
]
for s, r in zip(sentences, classifier(sentences)):
    print(f"{r['label']:9s} {r['score']:.4f}  | {s}")