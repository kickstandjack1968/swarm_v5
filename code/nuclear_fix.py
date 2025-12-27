import sys
import re

FILENAME = "src/swarm_coordinator_v2.py"

def fix_file():
    print(f"🔧 Scrubbing {FILENAME}...")
    
    with open(FILENAME, 'r') as f:
        lines = f.readlines()
        
    new_lines = []
    
    for i, line in enumerate(lines):
        # 1. Convert ALL tabs to 4 spaces
        clean_line = line.replace('\t', '    ')
        
        # 2. Strip trailing whitespace (invisible ghost spaces)
        clean_line = clean_line.rstrip() + '\n'
        
        # 3. FORCE alignment for the problem method
        # If this line defines _extract_exports_from_code, force it to exactly 4 spaces
        if "def _extract_exports_from_code(self," in clean_line:
            print(f"   -> Forcing alignment on line {i+1}")
            clean_line = "    def _extract_exports_from_code(self, code: str) -> List[str]:\n"
            
        new_lines.append(clean_line)
        
    with open(FILENAME, 'w') as f:
        f.writelines(new_lines)
        
    print("✅ File scrubbed successfully.")

if __name__ == "__main__":
    fix_file()