import os
import json
import subprocess
import shutil
import csv

# Load the JSON file with top 50 repositories
json_file = "mypy-dependents-by-stars.json"
with open(json_file, "r") as file:
    data = json.load(file)

# Extract top 50 repositories
top_50_repos = data[:50]

# Directory to clone repos
clone_dir = "cloned_repos"
os.makedirs(clone_dir, exist_ok=True)

# CSV file to store results
csv_file = "repo_analysis.csv"

# Function to clone, count Python files, and calculate size
def analyze_repo(author, repo, repo_url):
    repo_path = os.path.join(clone_dir, repo)

    # Clone the repository if not already cloned
    if not os.path.exists(repo_path):
        print(f"Cloning {repo}...")
        subprocess.run(["git", "clone", "--depth", "1", repo_url, repo_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Count Python files
    python_files = sum(len(files) for root, _, files in os.walk(repo_path) if any(f.endswith(".py") for f in files))

    # Calculate repo size in MB
    total_size = sum(os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(repo_path) for f in files)
    total_size_mb = total_size / (1024 * 1024)

    # Delete the cloned repo to save space
    #shutil.rmtree(repo_path, ignore_errors=True)

    return author, repo, repo_url, python_files, round(total_size_mb, 2)

# Process each repo and save results
with open(csv_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Author", "Repo", "URL", "Python Files", "Size (MB)"])

    for repo_data in top_50_repos:
        result = analyze_repo(repo_data["author"], repo_data["repo"], repo_data["repoUrl"])
        writer.writerow(result)
        print(f"✅ {result[1]} - {result[3]} Python files, {result[4]} MB")

print("\n🎉 Analysis complete! Results saved in 'repo_analysis.csv'")
