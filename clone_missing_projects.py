#!/usr/bin/env python3
"""
Script to automatically clone missing projects from GitHub at specific commits.
Matches project names from missing_invalid_files.json with repos in ManyTypes4PyDataset.txt
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Configuration
WORKSPACE_ROOT = Path(__file__).parent
MISSING_FILES_JSON = WORKSPACE_ROOT / "missing_invalid_files.json"
DATASET_FILE = WORKSPACE_ROOT / "ManyTypes4PyDataset.txt"
CLONE_DESTINATION = WORKSPACE_ROOT / "Missing_projects"
LOG_DIR = WORKSPACE_ROOT / "logs"

# Log files
SUCCESS_LOG = LOG_DIR / "success.log"
NOT_FOUND_LOG = LOG_DIR / "not_found.log"
COMMIT_FAILED_LOG = LOG_DIR / "commit_failed.log"
CLONE_FAILED_LOG = LOG_DIR / "clone_failed.log"
SUMMARY_LOG = LOG_DIR / "summary.json"


class ProjectCloner:
    def __init__(self):
        self.stats = {
            "total_missing_files": 0,
            "unique_projects": 0,
            "successfully_cloned": 0,
            "already_cloned": 0,
            "not_found_in_dataset": 0,
            "commit_not_found": 0,
            "clone_failed": 0,
        }
        self.processed_projects = set()
        self.repo_lookup = {}
        self.failed_projects = defaultdict(list)

    def setup_logging(self):
        """Create log directory"""
        LOG_DIR.mkdir(exist_ok=True)
        print(f"[INFO] Logs will be saved to: {LOG_DIR}")

    def parse_dataset(self):
        """Parse ManyTypes4PyDataset.txt into a lookup dictionary"""
        print("[INFO] Parsing ManyTypes4PyDataset.txt...")
        
        if not DATASET_FILE.exists():
            print(f"[ERROR] Dataset file not found: {DATASET_FILE}")
            sys.exit(1)

        with open(DATASET_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 2:
                    print(f"[WARN] Skipping malformed line: {line}")
                    continue
                
                url, commit = parts
                # Extract project name from URL (part before .git)
                # Example: https://github.com/dbt-labs/dbt.git -> dbt
                try:
                    project_name = url.split('/')[-1].replace('.git', '')
                    self.repo_lookup[project_name] = (url, commit)
                except Exception as e:
                    print(f"[WARN] Failed to parse URL: {url} - {e}")
                    continue
        
        print(f"[INFO] Loaded {len(self.repo_lookup)} projects from dataset")

    def extract_missing_projects(self):
        """Extract unique project names from missing_invalid_files.json"""
        print("[INFO] Extracting missing projects from JSON...")
        
        if not MISSING_FILES_JSON.exists():
            print(f"[ERROR] Missing files JSON not found: {MISSING_FILES_JSON}")
            sys.exit(1)

        projects = set()
        try:
            with open(MISSING_FILES_JSON, 'r') as f:
                data = json.load(f)
            
            for entry in data:
                target_path = entry.get("target_path", "")
                # Extract project name: first folder after "cloned_repos\"
                if "cloned_repos" in target_path:
                    parts = target_path.split("cloned_repos")[-1].strip("\\").split("\\")
                    if parts and parts[0]:
                        project_name = parts[0]
                        projects.add(project_name)
            
            self.stats["total_missing_files"] = len(data)
            self.stats["unique_projects"] = len(projects)
            print(f"[INFO] Found {len(projects)} unique projects from {len(data)} missing files")
            return projects
        
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse JSON: {e}")
            sys.exit(1)

    def match_project(self, project_name):
        """Match project name with repository in lookup - exact match only"""
        if project_name in self.repo_lookup:
            return project_name
        return None

    def clone_project(self, project_name, repo_key):
        """Clone project at specific commit"""
        url, commit = self.repo_lookup[repo_key]
        dest_path = CLONE_DESTINATION / project_name
        
        # Check if already cloned
        if dest_path.exists():
            print(f"[SKIP] Project already cloned: {project_name}")
            self.stats["already_cloned"] += 1
            return True
        
        try:
            print(f"[CLONE] Cloning {project_name}...")
            print(f"        URL: {url}")
            print(f"        Commit: {commit}")
            print(f"        Destination: {dest_path}")
            
            # Clone the repository
            result = subprocess.run(
                ["git", "clone", url, str(dest_path)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                error_msg = f"Clone failed for {project_name}: {result.stderr}"
                print(f"[ERROR] {error_msg}")
                self.failed_projects["clone_failed"].append((project_name, error_msg))
                self.stats["clone_failed"] += 1
                return False
            
            # Checkout specific commit
            result = subprocess.run(
                ["git", "checkout", commit],
                cwd=str(dest_path),
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                error_msg = f"Commit checkout failed for {project_name} at commit {commit}: {result.stderr}"
                print(f"[ERROR] {error_msg}")
                self.failed_projects["commit_failed"].append((project_name, url, commit, result.stderr))
                self.stats["commit_not_found"] += 1
                return False
            
            print(f"[SUCCESS] Successfully cloned {project_name} at commit {commit}")
            self.stats["successfully_cloned"] += 1
            return True
        
        except subprocess.TimeoutExpired:
            error_msg = f"Clone timeout for {project_name}"
            print(f"[ERROR] {error_msg}")
            self.failed_projects["clone_failed"].append((project_name, error_msg))
            self.stats["clone_failed"] += 1
            return False
        
        except Exception as e:
            error_msg = f"Unexpected error cloning {project_name}: {str(e)}"
            print(f"[ERROR] {error_msg}")
            self.failed_projects["clone_failed"].append((project_name, error_msg))
            self.stats["clone_failed"] += 1
            return False

    def write_logs(self):
        """Write all log files"""
        print("[INFO] Writing log files...")
        
        # Success log
        with open(SUCCESS_LOG, 'w') as f:
            f.write(f"Successfully cloned projects ({self.stats['successfully_cloned']})\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            for project_name in sorted(self.processed_projects):
                if project_name in self.repo_lookup or any(
                    p == project_name for p in [self.match_project(pn) for pn in self.processed_projects]
                ):
                    if project_name not in self.failed_projects.get("not_found", []) and \
                       project_name not in [p[0] for p in self.failed_projects.get("commit_failed", [])] and \
                       project_name not in [p[0] for p in self.failed_projects.get("clone_failed", [])]:
                        url, commit = self.repo_lookup.get(project_name, ("", ""))
                        f.write(f"{project_name}\n")
                        f.write(f"  URL: {url}\n")
                        f.write(f"  Commit: {commit}\n")
                        f.write(f"  Path: {CLONE_DESTINATION / project_name}\n\n")
        
        # Not found log
        with open(NOT_FOUND_LOG, 'w') as f:
            f.write(f"Projects not found in dataset ({self.stats['not_found_in_dataset']})\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            for project_name, reason in self.failed_projects.get("not_found", []):
                f.write(f"{project_name}\n  Reason: {reason}\n\n")
        
        # Commit failed log
        with open(COMMIT_FAILED_LOG, 'w') as f:
            f.write(f"Projects where commit checkout failed ({self.stats['commit_not_found']})\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            for project_name, url, commit, error in self.failed_projects.get("commit_failed", []):
                f.write(f"{project_name}\n")
                f.write(f"  URL: {url}\n")
                f.write(f"  Commit: {commit}\n")
                f.write(f"  Error: {error}\n\n")
        
        # Clone failed log
        with open(CLONE_FAILED_LOG, 'w') as f:
            f.write(f"Projects where cloning failed ({self.stats['clone_failed']})\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            for project_name, error in self.failed_projects.get("clone_failed", []):
                f.write(f"{project_name}\n  Error: {error}\n\n")
        
        # Summary
        with open(SUMMARY_LOG, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "stats": self.stats,
                "log_files": {
                    "success": str(SUCCESS_LOG),
                    "not_found": str(NOT_FOUND_LOG),
                    "commit_failed": str(COMMIT_FAILED_LOG),
                    "clone_failed": str(CLONE_FAILED_LOG),
                }
            }, f, indent=2)

    def run(self):
        """Main execution"""
        print("=" * 80)
        print("Missing Projects Cloner")
        print("=" * 80)
        
        self.setup_logging()
        self.parse_dataset()
        missing_projects = self.extract_missing_projects()
        
        # Create destination directory
        CLONE_DESTINATION.mkdir(exist_ok=True)
        print(f"[INFO] Clones will be saved to: {CLONE_DESTINATION}\n")
        
        print("[INFO] Processing projects...")
        print("-" * 80)
        
        for project_name in sorted(missing_projects):
            if project_name in self.processed_projects:
                continue
            
            self.processed_projects.add(project_name)
            
            # Try to match project
            repo_key = self.match_project(project_name)
            
            if not repo_key:
                msg = f"Project '{project_name}' not found in dataset"
                print(f"[NOT FOUND] {msg}")
                self.failed_projects["not_found"].append((project_name, msg))
                self.stats["not_found_in_dataset"] += 1
                continue
            
            # Clone the project
            self.clone_project(project_name, repo_key)
        
        print("-" * 80)
        print("[INFO] Writing logs...")
        self.write_logs()
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total missing files: {self.stats['total_missing_files']}")
        print(f"Unique projects: {self.stats['unique_projects']}")
        print(f"Successfully cloned: {self.stats['successfully_cloned']}")
        print(f"Already cloned: {self.stats['already_cloned']}")
        print(f"Not found in dataset: {self.stats['not_found_in_dataset']}")
        print(f"Commit not found: {self.stats['commit_not_found']}")
        print(f"Clone failed: {self.stats['clone_failed']}")
        print("\nLogs saved to:")
        print(f"  - {SUCCESS_LOG}")
        print(f"  - {NOT_FOUND_LOG}")
        print(f"  - {COMMIT_FAILED_LOG}")
        print(f"  - {CLONE_FAILED_LOG}")
        print(f"  - {SUMMARY_LOG}")
        print("=" * 80)


if __name__ == "__main__":
    cloner = ProjectCloner()
    cloner.run()
