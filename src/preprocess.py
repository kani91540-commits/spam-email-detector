import pandas as pd
import re

# Load the dataset
data = pd.read_csv("data/spam.csv")

print("Original Dataset:")
print(data.head())

print("\nOriginal Shape:")
print(data.shape)

# -----------------------------------
# 1. Check missing values
# -----------------------------------

print("\nMissing Values:")
print(data.isnull().sum())

# Remove rows with missing values
data = data.dropna()

# -----------------------------------
# 2. Remove duplicate messages
# -----------------------------------

print("\nDuplicate Rows Before Removing:")
print(data.duplicated().sum())

data = data.drop_duplicates()

print("Duplicate Rows After Removing:")
print(data.duplicated().sum())

# -----------------------------------
# 3. Convert messages to lowercase
# -----------------------------------

data["message"] = data["message"].str.lower()

# -----------------------------------
# 4. Remove punctuation and special characters
# -----------------------------------

def clean_text(text):

    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove spaces at beginning and end
    text = text.strip()

    return text


data["message"] = data["message"].apply(clean_text)

# -----------------------------------
# 5. Calculate message length
# -----------------------------------

data["message_length"] = data["message"].str.len()

# -----------------------------------
# Display cleaned data
# -----------------------------------

print("\nCleaned Dataset:")
print(data.head())

print("\nCleaned Shape:")
print(data.shape)

# -----------------------------------
# Save cleaned dataset
# -----------------------------------

data.to_csv("data/cleaned_spam.csv", index=False)

print("\nCleaned dataset saved successfully!")