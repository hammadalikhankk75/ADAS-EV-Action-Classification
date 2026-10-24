
import pandas as pd

# Use only the D: drive path
file_path = r"D:\ADAS-EV-Action-Classification\data\ADAS_EV_Dataset.csv"

print("File path:", file_path)

# Load the dataset ONCE
df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print(df.head())
