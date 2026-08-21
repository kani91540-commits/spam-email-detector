import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the dataset
data = pd.read_csv("data/spam.csv")

# 2. Display first 5 rows
print("First 5 rows:")
print(data.head())

# 3. Display last 5 rows
print("\nLast 5 rows:")
print(data.tail())

# 4. Display dataset shape
print("\nDataset Shape:")
print(data.shape)

# 5. Display column names
print("\nColumn Names:")
print(data.columns)

# 6. Display dataset information
print("\nDataset Information:")
data.info()

# 7. Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# 8. Check duplicate rows
print("\nDuplicate Rows:")
print(data.duplicated().sum())

# 9. Count spam and ham messages
print("\nMessage Distribution:")
print(data["label"].value_counts())

# 10. Calculate percentage of spam and ham
print("\nMessage Percentage:")
print(data["label"].value_counts(normalize=True) * 100)

# 11. Calculate message length
data["message_length"] = data["message"].str.len()

print("\nMessage Length Statistics:")
print(data["message_length"].describe())

# 12. Display first 5 rows with message length
print("\nData with Message Length:")
print(data.head())

# 13. Plot Spam vs Ham
data["label"].value_counts().plot(kind="bar")

plt.title("Spam vs Ham Messages")
plt.xlabel("Message Type")
plt.ylabel("Number of Messages")
plt.tight_layout()
plt.show()