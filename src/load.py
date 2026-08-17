import pandas as pd

# Load the SMS dataset
data = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Display first 5 rows
print("First 5 rows:")
print(data.head())

# Display dataset size
print("\nDataset Shape:")
print(data.shape)

# Count spam and normal messages
print("\nMessage Counts:")
print(data["label"].value_counts())

# Convert to CSV
data.to_csv("data/spam.csv", index=False)

print("\nspam.csv created successfully!")