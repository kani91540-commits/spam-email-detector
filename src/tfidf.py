import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load cleaned dataset
data = pd.read_csv("data/cleaned_spam.csv")

# Handle missing messages
data["message"] = data["message"].fillna("")

# Separate messages and labels
messages = data["message"]
labels = data["label"]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Convert text into numbers
X = vectorizer.fit_transform(messages)

# Display information
print("TF-IDF conversion completed!")

print("\nOriginal number of messages:")
print(len(messages))

print("\nNumber of TF-IDF features:")
print(len(vectorizer.get_feature_names_out()))

print("\nTF-IDF Matrix Shape:")
print(X.shape)

# Display some words
print("\nFirst 20 TF-IDF Features:")
print(vectorizer.get_feature_names_out()[:20])

# Display first message
print("\nFirst Message:")
print(messages.iloc[0])

# Display first message as TF-IDF values
print("\nTF-IDF values of first message:")
print(X[0].toarray())

print("\nDay 6 completed successfully!")
