# Exercise 4.18 - NLTK dengan teks berbeda
# pip install nltk
import nltk
nltk.download('punkt'); nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker'); nltk.download('maxent_ne_chunker_tab'); nltk.download('words')

sentence = """
Politeknik Elektronika Negeri Surabaya is a vocational university located in
Surabaya, Indonesia. Google released the Android operating system in 2008,
and Elon Musk founded SpaceX in California. Students at PENS build multimedia
applications using Python and TensorFlow.
"""
tokens = nltk.word_tokenize(sentence)
print(tokens)
tagged = nltk.pos_tag(tokens)
print(tagged)
entities = nltk.chunk.ne_chunk(tagged)
print(entities)