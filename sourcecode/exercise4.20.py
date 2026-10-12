# Exercise 4.20 - Question answering dengan teks & pertanyaan berbeda
# pip install transformers torch
import os
os.environ["USE_TF"] = "0"
from transformers import pipeline
qa = pipeline('question-answering')

context = ("Politeknik Elektronika Negeri Surabaya (PENS) was founded in 1987 and is "
           "located in Surabaya, East Java, Indonesia. The Multimedia Engineering "
           "Technology program teaches students to build games, mixed reality "
           "applications, and web systems.")

questions = [
    "When was PENS founded?",
    "Where is PENS located?",
    "What does the Multimedia Engineering Technology program teach?",
    "Who is the rector of PENS?",       # jawaban tidak ada di teks
]
for q in questions:
    r = qa({'question': q, 'context': context})
    print(f"Q: {q}\n   A: {r['answer']!r}  (score={r['score']:.3f}, start={r['start']}, end={r['end']})")