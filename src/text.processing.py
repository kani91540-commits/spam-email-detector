import pandas as pd

# Load cleaned dataset
data = pd.read_csv("data/cleaned_spam.csv")

print("Original Cleaned Messages:")
print(data["message"].head())


# -----------------------------------
# 1. Remove missing messages
# -----------------------------------

data["message"] = data["message"].fillna("")


# -----------------------------------
# 2. Function to tokenize text
# -----------------------------------

def tokenize_text(text):
    return text.split()


# -----------------------------------
# 3. Apply tokenization
# -----------------------------------

data["tokens"] = data["message"].apply(tokenize_text)


# -----------------------------------
# 4. Display tokenized messages
# -----------------------------------

print("\nTokenized Messages:")
print(data[["message", "tokens"]].head())


# -----------------------------------
# 5. Count number of words
# -----------------------------------

data["word_count"] = data["tokens"].apply(len)

print("\nWord Count:")
print(data[["message", "word_count"]].head())


# -----------------------------------
# 6. Calculate average word count
# -----------------------------------

print("\nAverage Word Count:")
print(data["word_count"].mean())


# -----------------------------------
# 7. Save processed dataset
# -----------------------------------

data.to_csv("data/processed_spam.csv", index=False)

print("\nProcessed dataset saved successfully!")