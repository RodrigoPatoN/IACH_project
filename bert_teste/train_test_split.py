import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file
data = pd.read_csv('./datasets/data_processed.csv')

# Split the data into training and testing subsets
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# Save the training and testing subsets to separate files
train_data.to_csv('./datasets/train_data.csv', index=False)
test_data.to_csv('./datasets/test_data.csv', index=False)
