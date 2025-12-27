import os
import re

FILE_PATH = "src/swarm_coordinator_v2.py"

def fix_file():
    if not os.path.exists(FILE_PATH):
        print(f"Error: Could not find {FILE_PATH}")
        return

    print(f"Reading {FILE_PATH}...")
    with open(FILE_PATH, 'r') as f:
        lines = f.readlines()

    new_lines = []
    
    # 1. Normalize Indentation (Convert Tabs to Spaces)
    print("Normalizing indentation (Tabs -> 4 Spaces)...")
    for line in lines:
        # Replace explicit tabs with 4 spaces
        clean_line = line.replace('\t', '    ')
        new_lines.append(clean_line)

    # 2. Fix the specific transition from execute_task to _extract_exports
    print("Fixing method transition...")
    final_lines = []
    inside_execute_task = False
    
    for i, line in enumerate(new_lines):
        stripped = line.strip()
        
        # Detect start of execute_task
        if "def execute_task(self, task: Task)" in line:
            inside_execute_task = True
            final_lines.append(line)
            continue

        # Detect the problematic helper method definition
        if "def _extract_exports_from_code(self," in line:
            if inside_execute_task:
                # We found the start of the next function.
                # Ensure the previous lines (the end of execute_task) are closed properly.
                
                # Check previous non-empty line
                prev_idx = len(final_lines) - 1
                while prev_idx >= 0 and not final_lines[prev_idx].strip():
                    prev_idx -= 1
                
                # If the previous logic block looks like it's inside an 'except', 
                # ensure we have a clean break.
                inside_execute_task = False
                
                # Force alignment of this line to 4 spaces (Class method level)
                final_lines.append("    def _extract_exports_from_code(self, code: str) -> List[str]:\n")
                continue
        
        # If we are just appending lines, ensure specific ones are valid
        final_lines.append(line)

    # 3. Write back
    with open(FILE_PATH, 'w') as f:
        f.writelines(final_lines)
    
    print("✅ File fixed! Indentation normalized.")

if __name__ == "__main__":
    fix_file()