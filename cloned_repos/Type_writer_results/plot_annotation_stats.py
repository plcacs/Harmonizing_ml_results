import json
import matplotlib.pyplot as plt
import numpy as np

def get_annotation_stats(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Get annotation differences for files with score 0
    annotation_diffs = [info.get('original_parameters_with_annotations', 0) - info.get('updated_parameters_with_annotations', 0) 
                       for info in data.values() if info.get('score', 0) == 0]
    return annotation_diffs

# Load data from all files
files = ['gpt4O_stats_equal.json', 'o1_mini_stats_equal.json','deepseek_stats_equal.json' ]
model_names = ['GPT-4', 'O1-Mini','DeepSeek']

# Define bins
bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, float('inf')]
bin_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11-15', '16-20', '>20']

# Create figure
plt.figure(figsize=(20, 8))

# Plot bars for each model
bar_width = 0.25

# Get all data first to determine non-empty bins
all_diffs = []
for file in files:
    all_diffs.extend(get_annotation_stats(file))

# Calculate histogram for all data to find non-empty bins
hist_all, _ = np.histogram(all_diffs, bins=bins)
non_empty_bins = hist_all > 0

# Create x positions only for non-empty bins
x = np.arange(np.sum(non_empty_bins))

for i, (file, name) in enumerate(zip(files, model_names)):
    diffs = get_annotation_stats(file)
    hist, _ = np.histogram(diffs, bins=bins)
    plt.bar(x + i*bar_width, hist[non_empty_bins], bar_width, label=name)

plt.xlabel('Number of Annotations Removed', fontsize=24)
plt.ylabel('Number of Files with Score 0', fontsize=24 )
plt.title('Distribution of Files with Score 0 by Number of Annotations Removed (ManyTypes4Py)', fontsize=24)
plt.xticks(x + bar_width, np.array(bin_labels[:-1])[non_empty_bins], rotation=45, fontsize=24)
plt.yticks(fontsize=24)
plt.legend()

plt.tight_layout()
#plt.show()
plt.savefig('annotation_removal_stats_ManyTypes4Py.pdf',bbox_inches='tight')
plt.close() 