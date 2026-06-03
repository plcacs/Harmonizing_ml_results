#!/usr/bin/env python3
"""
Verification script to check if missing files exist in the cloned projects.
Simple direct path matching approach.
"""

import json
from pathlib import Path
from datetime import datetime

# Configuration
WORKSPACE_ROOT = Path(__file__).parent
MISSING_FILES_JSON = WORKSPACE_ROOT / "missing_invalid_files.json"
CLONE_DESTINATION = WORKSPACE_ROOT / "Missing_projects"
LOG_DIR = WORKSPACE_ROOT / "verification_logs"

# Log files
FILES_FOUND_LOG = LOG_DIR / "files_found.log"
FILES_NOT_FOUND_LOG = LOG_DIR / "files_not_found.log"
VERIFICATION_SUMMARY = LOG_DIR / "verification_summary.json"


class FilesVerifier:
    def __init__(self):
        self.stats = {
            "total_missing_files": 0,
            "files_found": 0,
            "files_not_found": 0,
        }
        self.found_files = []
        self.not_found_files = []

    def setup_logging(self):
        """Ensure log directory exists"""
        LOG_DIR.mkdir(exist_ok=True)
        print(f"[INFO] Verification logs will be saved to: {LOG_DIR}")

    def extract_relative_path(self, target_path):
        """
        Extract the path after 'cloned_repos' and convert to Unix format.
        Example: 
        Input:  D:\r\HarmonizingML\HarmonizingML\code\cloned_repos\dbt\core\dbt\task\run.py
        Output: dbt/core/dbt/task/run.py
        """
        if "cloned_repos" not in target_path:
            return None
        
        # Split by cloned_repos and get everything after it
        after_cloned_repos = target_path.split("cloned_repos")[-1]
        
        # Strip leading backslashes
        after_cloned_repos = after_cloned_repos.lstrip("\\").lstrip("/")
        
        # Convert Windows path to Unix path
        unix_path = after_cloned_repos.replace("\\", "/")
        
        return unix_path

    def verify_file(self, target_path, alias_name):
        """
        Verify if file exists by checking the exact Unix path in Missing_projects.
        """
        relative_path_unix = self.extract_relative_path(target_path)
        
        if not relative_path_unix:
            return False, None
        
        # Construct the full path
        file_path = CLONE_DESTINATION / relative_path_unix
        
        # Check if file exists
        exists = file_path.exists()
        
        return exists, str(file_path)

    def run(self):
        """Main execution"""
        print("=" * 80)
        print("Missing Files Verifier (Direct Path Matching)")
        print("=" * 80)
        
        self.setup_logging()
        
        if not MISSING_FILES_JSON.exists():
            print(f"[ERROR] Missing files JSON not found: {MISSING_FILES_JSON}")
            return
        
        if not CLONE_DESTINATION.exists():
            print(f"[ERROR] Missing_projects folder not found: {CLONE_DESTINATION}")
            return
        
        print(f"[INFO] Reading {MISSING_FILES_JSON}...")
        print(f"[INFO] Checking against {CLONE_DESTINATION}...\n")
        print("[INFO] Processing files...")
        print("-" * 80)
        
        try:
            with open(MISSING_FILES_JSON, 'r') as f:
                data = json.load(f)
            
            self.stats["total_missing_files"] = len(data)
            
            for idx, entry in enumerate(data, 1):
                alias_name = entry.get("alias_name", "unknown")
                target_path = entry.get("target_path", "")
                
                if not target_path:
                    continue
                
                exists, actual_path = self.verify_file(target_path, alias_name)
                
                if exists:
                    self.stats["files_found"] += 1
                    self.found_files.append({
                        "alias_name": alias_name,
                        "target_path": target_path,
                        "found_at": actual_path,
                    })
                    print(f"[✓ {idx:3d}] FOUND: {alias_name}")
                else:
                    self.stats["files_not_found"] += 1
                    self.not_found_files.append({
                        "alias_name": alias_name,
                        "target_path": target_path,
                        "expected_path": actual_path,
                    })
                    print(f"[✗ {idx:3d}] NOT_FOUND: {alias_name}")
            
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse JSON: {e}")
            return
        
        print("-" * 80)
        print("[INFO] Writing log files...")
        self.write_logs()
        
        print("\n" + "=" * 80)
        print("VERIFICATION SUMMARY")
        print("=" * 80)
        print(f"Total missing files checked: {self.stats['total_missing_files']}")
        print(f"Files found: {self.stats['files_found']} ✓")
        print(f"Files not found: {self.stats['files_not_found']} ✗")
        
        if self.stats['total_missing_files'] > 0:
            success_rate = (self.stats['files_found'] / self.stats['total_missing_files']) * 100
            print(f"Success rate: {success_rate:.1f}%")
        
        print("\nLogs saved to:")
        print(f"  - {FILES_FOUND_LOG}")
        print(f"  - {FILES_NOT_FOUND_LOG}")
        print(f"  - {VERIFICATION_SUMMARY}")
        print("=" * 80)

    def write_logs(self):
        """Write all log files"""
        
        # Files found log
        with open(FILES_FOUND_LOG, 'w') as f:
            f.write(f"Successfully Found Files ({self.stats['files_found']})\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            for file_info in self.found_files:
                f.write(f"Alias: {file_info['alias_name']}\n")
                f.write(f"  Original target path: {file_info['target_path']}\n")
                f.write(f"  Found at: {file_info['found_at']}\n\n")
        
        # Files not found log
        with open(FILES_NOT_FOUND_LOG, 'w') as f:
            f.write(f"Missing Files ({self.stats['files_not_found']})\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            for file_info in self.not_found_files:
                f.write(f"Alias: {file_info['alias_name']}\n")
                f.write(f"  Original target path: {file_info['target_path']}\n")
                f.write(f"  Expected path: {file_info['expected_path']}\n\n")
        
        # Summary
        with open(VERIFICATION_SUMMARY, 'w') as f:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "stats": self.stats,
            }
            if self.stats['total_missing_files'] > 0:
                summary["success_rate_percent"] = round(
                    self.stats['files_found'] / self.stats['total_missing_files'] * 100, 1
                )
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    verifier = FilesVerifier()
    verifier.run()
