from goldman_emma_raw import goldman_docs
from henson_matthew_raw import henson_docs
from wu_tingfang_raw import wu_docs
# import sklearn modules here:
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB 

# Setting up the combined list of friends' writing samples
friends_docs = goldman_docs + henson_docs + wu_docs
# We are assigning each friend a tag, of 1, 2, 3. The magic numbers are in reference to the length of each doc.
friends_labels = [1] * 154 + [2] * 141 + [3] * 166


mystery_postcard = """
My friend,
From the 10th of July to the 13th, a fierce storm raged, clouds of
freeing spray broke over the ship, incasing her in a coat of icy mail,
and the tempest forced all of the ice out of the lower end of the
channel and beyond as far as the eye could see, but the _Roosevelt_
still remained surrounded by ice.
Hope to see you soon.
"""


bow_vectorizer = CountVectorizer()

# We are a creating a feature dictionary mapping, of every word in the combined friends docs
bow_vectorizer.fit(friends_docs)
# We now count the occurance of each word in friends_doc, ie. vectorization
friends_vectors = bow_vectorizer.transform(friends_docs)

# We count the occurance of each word (in relation to the friends_docs dictionary) in the postcard
mystery_vector = bow_vectorizer.transform([mystery_postcard])

# Implementing a Naive Bayes classifier
friends_classifer = MultinomialNB()


friends_classifer.fit(friends_vectors, friends_labels)

predictions = friends_classifer.predict(mystery_vector)

mystery_friend = predictions[0] if predictions[0] else "someone else"

print("The postcard was from {}!".format(mystery_friend))