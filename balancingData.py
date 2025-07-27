import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from extrationAndLabel import extract_blocks_with_features, label_pdf_blocks

directory = '/content/drive/MyDrive/Adobe_Hackathon/input'
output_path = '/content/real_output.json'
output = []

for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    # print(filepath)
    output_filepath = filepath.replace("input","output").replace(".pdf",".json")
    features = extract_blocks_with_features(filepath)
    labeled_features = label_pdf_blocks(features, output_filepath)
    output+=(labeled_features)


#Filitering words less thean 4 chars

real_output = []
print(f"len of output is {len(output)}")
for item in output:
  # print(item.get("text","")+"\n\n-----")
  if len(item.get("text",""))>4:
    real_output.append(item)


with open(output_path, "w", encoding="utf-8") as f:
        json.dump(real_output, f, indent=2, ensure_ascii=False)

print(f"len of output is {len(real_output)}")

print(f"\n✅ Labeled features saved to {output_path}")
print("\n📋 Preview (first 10 lines):\n")
# real_output[:10]



# Load the JSON file
with open('real_output.json', 'r') as f:
    data = json.load(f)

# Flatten the nested list and extract labels
labels = []
for item in data:
    labels.append(item['label'])

# Create a pandas Series for easy value counting
df = pd.Series(labels)

# Count the occurrences of each label
label_counts = df.value_counts()

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Labels')
plt.ylabel('')  # Hide the y-label
plt.savefig('labels_pie_chart.png')

print("Pie chart saved as labels_pie_chart.png")
print(label_counts)




# Load the JSON file
# Ensure 'real_output.json' is in the same directory as this script
try:
    with open('real_output.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print("Error: 'real_output.json' not found. Please make sure the file is in the correct directory.")
    
# --- Data Balancing ---
# Convert the list of dictionaries to a pandas DataFrame for easier manipulation
df = pd.DataFrame(data)

# Separate the majority class (label 0) from the minority classes (all other labels)
df_majority = df[df['label'] == 0]
df_minority = df[df['label'] != 0]

# Check if there are minority samples to balance against
if not df_minority.empty and not df_majority.empty:
    # Downsample the majority class.
    # We'll select a random sample from the majority class that is equal in size
    # to the total number of all minority classes.
    # Using random_state ensures that the sampling is reproducible.
    df_majority_downsampled = df_majority.sample(n=len(df_minority), random_state=42)

    # Combine the downsampled majority class with the original minority classes
    # to create a new, balanced DataFrame.
    df_balanced = pd.concat([df_majority_downsampled, df_minority])

    print("Dataset has been balanced.")
    print(f"Original size of label '0': {len(df_majority)}")
    print(f"Total size of other labels: {len(df_minority)}")
    print(f"New size of label '0' after downsampling: {len(df_majority_downsampled)}")

else:
    # If there are no minority or majority samples, use the original dataframe
    df_balanced = df
    print("Dataset does not require balancing or is empty.")


# --- Visualization ---
# Count the occurrences of each label in the new balanced dataset
# We use the 'label' column from our balanced DataFrame.
label_counts = df_balanced['label'].value_counts()

# Create a pie chart from the balanced data
plt.figure(figsize=(10, 10))
pie_wedges, _, _ = plt.pie(
    label_counts,
    autopct='%1.1f%%',
    startangle=140,
    textprops={'fontsize': 12, 'color': 'white', 'fontweight': 'bold'}
)
plt.title('Distribution of Labels (Balanced Dataset)', fontsize=16)
plt.legend(pie_wedges, label_counts.index, title="Labels", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.ylabel('')  # Hide the y-label for a cleaner look
plt.tight_layout()

# Save the new pie chart to a different file to avoid overwriting the original
output_filename = 'labels_pie_chart_balanced.png'
plt.savefig(output_filename)

print(f"\nBalanced pie chart saved as {output_filename}")
print("\nNew Label Counts:")
print(label_counts)
