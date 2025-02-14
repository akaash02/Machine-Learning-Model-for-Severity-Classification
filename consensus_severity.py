import pandas as pd
from collections import Counter

# Load deficiency dataset
from google.colab import files
uploaded = files.upload()

# Load the dataset
data = pd.read_csv('psc_severity_train.csv')

print(data.head())
print(data.columns)

df = pd.DataFrame(data)

# Replace newline characters in all string columns of the DataFrame
for column in df.select_dtypes(include=['object']).columns:
    df[column] = df[column].str.replace('\n', ' ', regex=False)

# Display the cleaned DataFrame
print(df)

# Fill missing values in categorical columns with a placeholder string
df['annotation_severity'] = df['annotation_severity'].fillna('Unknown')

# Check that there are no more missing values
print(df.isnull().sum())

df = df[~df['annotation_severity'].str.lower().isin(['unknown', 'not a deficiency'])]

# Count the rows where the condition is True
count_invalid = df['annotation_severity'].str.lower().isin(['unknown', 'not a deficiency']).sum()

# If you want to print a message:
if count_invalid > 0:
    print(f"There are {count_invalid} rows with 'Unknown' or 'Not a deficiency'.")
else:
    print("No rows with 'Unknown' or 'Not a deficiency' found.")

# Define the mapping for severity labels
severity_map = {'Low': 1, 'Medium': 2, 'High': 3}

# Apply the mapping to the 'annotation_severity' column
df['severity_score'] = df['annotation_severity'].map(severity_map)

# Group by 'PscInspectionId' and calculate the average severity score
grouped_df = df.groupby('PscInspectionId').agg(
    def_text=('def_text', 'first'),                   # Take the first 'def_text' for each group
    VesselGroup=('VesselGroup', 'first'),             # Take the first 'VesselGroup' for each group
    age=('age', 'mean'),                              # Take the mean of 'age' if numeric
    deficiency_code=('deficiency_code', 'first'),     # Keep original headers
    annotation_id=('annotation_id', 'first'),
    username=('username', 'first'),
    annotation_severity=('annotation_severity', 'first'),
    InspectionDate=('InspectionDate', 'first'),
    VesselId=('VesselId', 'first'),
    PscAuthorityId=('PscAuthorityId', 'first'),
    PortId=('PortId', 'first'),
    avg_severity_score=('severity_score', 'mean')     # Calculate average severity score
).reset_index()

# Map the average severity score back to the severity label
def map_to_severity(score):
    if score <= 1.49:
        return 'Low'
    elif score <= 2.49:
        return 'Medium'
    else:
        return 'High'

# Apply the mapping function to create the 'consensus_severity' column
grouped_df['consensus_severity'] = grouped_df['avg_severity_score'].apply(map_to_severity)

# Display the final DataFrame with all original headers except 'severity_score' and the new columns
print(grouped_df)

# Save the consensus severity results to a CSV
grouped_df.to_csv('cleaned_data.csv', index=False)
files.download('cleaned_data.csv')
