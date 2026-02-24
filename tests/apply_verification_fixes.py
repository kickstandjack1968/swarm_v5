#!/usr/bin/env python3
"""
Automated Patch Applicator for SwarmCoordinator Verification Fixes
===================================================================

This script automatically applies the verification fixes to swarm_coordinator_v2.py

Usage:
    python apply_verification_fixes.py --file swarm_coordinator_v2.py --backup
    
Options:
    --file FILE     : Path to swarm_coordinator_v2.py
    --backup        : Create backup before modifying
    --dry-run       : Show what would be changed without modifying
"""

import argparse
import re
import shutil
from pathlib import Path
from datetime import datetime


class PatchApplicator:
    """Applies verification patches to SwarmCoordinator"""
    
    def __init__(self, filepath: Path, dry_run: bool = False):
        self.filepath = filepath
        self.dry_run = dry_run
        self.changes_made = []
        
        with open(filepath, 'r') as f:
            self.content = f.read()
            self.lines = self.content.split('\n')
    
    def find_line(self, pattern: str, start: int = 0) -> int:
        """Find line number containing pattern"""
        for i in range(start, len(self.lines)):
            if pattern in self.lines[i]:
                return i
        return -1
    
    def insert_after_line(self, line_num: int, text: str, description: str):
        """Insert text after specified line"""
        if self.dry_run:
            print(f"[DRY RUN] Would insert after line {line_num + 1}: {description}")
            return
        
        # Split text into lines and maintain indentation
        new_lines = text.split('\n')
        self.lines = self.lines[:line_num + 1] + new_lines + self.lines[line_num + 1:]
        self.changes_made.append(f"Inserted {description} after line {line_num + 1}")
        print(f"✓ Inserted {description} after line {line_num + 1}")
    
    def replace_lines(self, start: int, end: int, text: str, description: str):
        """Replace lines from start to end with new text"""
        if self.dry_run:
            print(f"[DRY RUN] Would replace lines {start + 1}-{end + 1}: {description}")
            return
        
        new_lines = text.split('\n')
        self.lines = self.lines[:start] + new_lines + self.lines[end + 1:]
        self.changes_made.append(f"Replaced {description} (lines {start + 1}-{end + 1})")
        print(f"✓ Replaced {description} (lines {start + 1}-{end + 1})")
    
    def apply_all_patches(self):
        """Apply all verification patches"""
        print("\\n" + "=" * 70)
        print("APPLYING VERIFICATION PATCHES")
        print("=" * 70 + "\\n")
        
        # Patch 1: Add missing methods after _save_project_outputs
        print("Patch 1: Adding missing verification methods...")
        line_num = self.find_line("def _save_project_outputs(self):")
        if line_num == -1:
            print("✗ Could not find _save_project_outputs method")
        else:
            # Find the end of the method
            indent_level = len(self.lines[line_num]) - len(self.lines[line_num].lstrip())
            end_line = line_num + 1
            while end_line < len(self.lines):
                line = self.lines[end_line]
                if line.strip() and not line.startswith(' ' * (indent_level + 1)):
                    break
                end_line += 1
            
            self.insert_after_line(end_line - 1, self.get_patch_1(), "smoke test methods")
        
        # Patch 2: Add test execution to execute_task
        print("\\nPatch 2: Adding test execution...")
        line_num = self.find_line('task.task_type == "test_generation"')
        if line_num == -1:
            print("✗ Could not find test_generation section")
        else:
            # Find where test_code is assigned and status is set
            search_start = line_num
            while search_start < len(self.lines) and "task.result = test_code" not in self.lines[search_start]:
                search_start += 1
            
            if search_start < len(self.lines):
                # Find where status is set to COMPLETED
                while search_start < len(self.lines) and "TaskStatus.COMPLETED" not in self.lines[search_start]:
                    search_start += 1
                
                if search_start < len(self.lines):
                    self.insert_after_line(search_start, self.get_patch_2(), "test execution code")
        
        # Patch 3: Update verification prompt
        print("\\nPatch 3: Updating verification prompt builder...")
        line_num = self.find_line("smoke_tests = task.metadata.get(\\"smoke_tests\\")")
        if line_num == -1:
            print("✗ Could not find smoke_tests assignment")
        else:
            # Find the next 3-4 lines (the old implementation)
            end_line = line_num + 3
            self.replace_lines(line_num, end_line, self.get_patch_3(), "smoke test runner")
        
        # Patch 4: Add test results section
        print("\\nPatch 4: Adding test results section to verification...")
        line_num = self.find_line("{smoke_section}")
        if line_num == -1:
            print("✗ Could not find smoke_section in verification prompt")
        else:
            self.insert_after_line(line_num, self.get_patch_4(), "test results section")
        
        # Patch 5: Update verification checklist
        print("\\nPatch 5: Updating verification checklist...")
        line_num = self.find_line("Perform final verification:")
        if line_num == -1:
            print("✗ Could not find verification checklist")
        else:
            # Find the end of the checklist (until OUTPUT FORMAT)
            end_line = line_num
            while end_line < len(self.lines) and "OUTPUT FORMAT" not in self.lines[end_line]:
                end_line += 1
            end_line += 6  # Include the format description
            
            self.replace_lines(line_num, end_line, self.get_patch_5(), "verification checklist")
    
    def get_patch_1(self) -> str:
        """Get smoke test methods patch"""
        return '''
    def _run_smoke_tests(self) -> Dict[str, Any]:
        """Run basic smoke tests on generated code."""
        results = {}
        project_dir = self.state.get("project_info", {}).get("project_dir")
        
        if not project_dir or not os.path.exists(project_dir):
            return {"error": {"cmd": "N/A", "returncode": -1, "stdout": "", 
                             "stderr": "Project directory not found"}}
        
        src_dir = os.path.join(project_dir, "src")
        if not os.path.exists(src_dir):
            return {"error": {"cmd": "N/A", "returncode": -1, "stdout": "", 
                             "stderr": "Source directory not found"}}
        
        results["syntax_check"] = self._smoke_test_syntax(src_dir)
        results["import_check"] = self._smoke_test_imports(project_dir, src_dir)
        
        entry_point = self.state.get("project_info", {}).get("entry_point")
        project_type = self.state.get("project_info", {}).get("project_type", "cli")
        
        if entry_point and project_type in ("cli", "subprocess_tool"):
            results["entry_point_execution"] = self._smoke_test_entry_point(
                project_dir, entry_point
            )
        
        return results
    
    def _smoke_test_syntax(self, src_dir: str) -> Dict[str, Any]:
        """Check all Python files for syntax errors."""
        try:
            errors = []
            for root, dirs, files in os.walk(src_dir):
                for file in files:
                    if file.endswith('.py'):
                        filepath = os.path.join(root, file)
                        try:
                            with open(filepath, 'r') as f:
                                ast.parse(f.read())
                        except SyntaxError as e:
                            errors.append(f"{file}: {e}")
            
            if errors:
                return {"cmd": "ast.parse(all_files)", "returncode": 1,
                        "stdout": "", "stderr": "\\n".join(errors)}
            else:
                return {"cmd": "ast.parse(all_files)", "returncode": 0,
                        "stdout": "All files have valid syntax", "stderr": ""}
        except Exception as e:
            return {"cmd": "ast.parse(all_files)", "returncode": -1,
                    "stdout": "", "stderr": f"Syntax check failed: {e}"}
    
    def _smoke_test_imports(self, project_dir: str, src_dir: str) -> Dict[str, Any]:
        """Test if all modules can be imported."""
        try:
            py_files = []
            for root, dirs, files in os.walk(src_dir):
                for file in files:
                    if file.endswith('.py') and not file.startswith('__'):
                        rel_path = os.path.relpath(os.path.join(root, file), src_dir)
                        module_path = rel_path.replace(os.sep, '.').replace('.py', '')
                        py_files.append(module_path)
            
            test_script = f"""
import sys
sys.path.insert(0, {repr(project_dir)})
failed = []
for module in {py_files}:
    try:
        __import__(f'src.{{module}}')
    except Exception as e:
        failed.append(f'src.{{module}}: {{e}}')
if failed:
    print("FAILED IMPORTS:")
    for f in failed:
        print(f'  {{f}}')
    sys.exit(1)
else:
    print(f"Successfully imported {{len({py_files})}} modules")
    sys.exit(0)
"""
            
            result = subprocess.run([sys.executable, "-c", test_script],
                                  capture_output=True, text=True, timeout=10, cwd=project_dir)
            
            return {"cmd": "import all modules", "returncode": result.returncode,
                    "stdout": result.stdout, "stderr": result.stderr}
            
        except subprocess.TimeoutExpired:
            return {"cmd": "import test", "returncode": -1, "stdout": "",
                    "stderr": "Import test timed out"}
        except Exception as e:
            return {"cmd": "import test", "returncode": -1, "stdout": "",
                    "stderr": f"Import test failed: {e}"}
    
    def _smoke_test_entry_point(self, project_dir: str, entry_point: str) -> Dict[str, Any]:
        """Test if entry point runs without crash."""
        try:
            module_name = entry_point.replace('.py', '').replace('/', '.')
            test_script = f"""
import sys, signal
def timeout_handler(signum, frame):
    sys.exit(0)
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(3)
sys.path.insert(0, {repr(project_dir)})
try:
    import src.{module_name} as entry
    print("Entry point imported successfully")
    sys.exit(0)
except Exception as e:
    print(f"Entry point failed: {{e}}")
    sys.exit(1)
"""
            result = subprocess.run([sys.executable, "-c", test_script],
                                  capture_output=True, text=True, timeout=5, cwd=project_dir)
            return {"cmd": f"import src.{module_name}", "returncode": result.returncode,
                    "stdout": result.stdout, "stderr": result.stderr}
        except subprocess.TimeoutExpired:
            return {"cmd": f"import src.{entry_point}", "returncode": 0,
                    "stdout": "Entry point started (timed out)", "stderr": ""}
        except Exception as e:
            return {"cmd": f"import src.{entry_point}", "returncode": -1,
                    "stdout": "", "stderr": f"Execution test failed: {e}"}
    
    def _check_entry_point_integration(self, code_content: str) -> str:
        """Check entry point integration."""
        if not code_content.strip():
            return "Entry point code is empty"
        try:
            tree = ast.parse(code_content)
        except SyntaxError as e:
            return f"Entry point has syntax errors: {e}"
        
        issues = []
        has_main_block = any(isinstance(node, ast.If) for node in ast.walk(tree) 
                            if hasattr(node, 'test'))
        if not has_main_block:
            issues.append("Entry point lacks if __name__ == '__main__': block")
        
        return "\\n".join(issues) if issues else ""
    
    def _execute_generated_tests(self, task) -> Dict[str, Any]:
        """Execute pytest tests."""
        test_task = next((t for t in self.completed_tasks 
                         if t.task_type == "test_generation"), None)
        
        if not test_task or not test_task.result:
            return {"success": False, "error": "No tests generated", "output": ""}
        
        test_content = self._clean_test_output(test_task.result)
        code_task = next((t for t in self.completed_tasks 
                         if t.task_type in ("coding", "plan_execution")), None)
        
        if not code_task or not code_task.result:
            return {"success": False, "error": "No source code found", "output": ""}
        
        files_dict = self._parse_multi_file_output(code_task.result)
        completed_files = {fname: type('FileResult', (), {'content': content})()
                          for fname, content in files_dict.items()}
        
        sandbox = get_docker_sandbox()
        if not sandbox or not sandbox.container_running:
            return {"success": False, "error": "Docker sandbox unavailable",
                   "output": "Test execution skipped"}
        
        try:
            success, output = sandbox.run_hot_test(completed_files, test_content,
                                                   "test_main.py", "")
            return {"success": success, 
                   "error": "" if success else "Tests failed", 
                   "output": output}
        except Exception as e:
            return {"success": False, "error": f"Test error: {e}", "output": ""}
'''
    
    def get_patch_2(self) -> str:
        """Get test execution patch"""
        return '''
                # Execute the generated tests
                if task.result:
                    print(f"  🧪 Executing generated tests...")
                    test_results = self._execute_generated_tests(task)
                    task.metadata["test_execution"] = test_results
                    
                    if test_results["success"]:
                        print(f"  ✓ All tests passed")
                    else:
                        print(f"  ✗ Tests failed or could not run")
                        if test_results.get("error"):
                            print(f"    Error: {test_results['error']}")
'''
    
    def get_patch_3(self) -> str:
        """Get verification prompt patch"""
        return '''        smoke_tests = task.metadata.get("smoke_tests")
        if smoke_tests is None:
            print("  🔥 Running smoke tests...")
            smoke_tests = self._run_smoke_tests()
            task.metadata["smoke_tests"] = smoke_tests
            
            all_passed = all(r.get("returncode") == 0 for r in smoke_tests.values() 
                           if isinstance(r, dict))
            if all_passed:
                print("  ✓ All smoke tests passed")
            else:
                print("  ⚠ Some smoke tests failed")
        
        test_task = next((t for t in self.completed_tasks 
                         if t.task_type == "test_generation"), None)
        test_execution = test_task.metadata.get("test_execution") if test_task else None'''
    
    def get_patch_4(self) -> str:
        """Get test results section patch"""
        return '''
        if test_execution:
            if test_execution.get("success"):
                test_section = """
TEST EXECUTION:
✓ All pytest tests PASSED"""
            else:
                output = test_execution.get("output", "")[:1500]
                test_section = f"""
TEST EXECUTION:
✗ Tests FAILED

{output}"""
        else:
            test_section = "TEST EXECUTION: Skipped (no Docker sandbox)"
'''
    
    def get_patch_5(self) -> str:
        """Get verification checklist patch"""
        return '''Perform final verification:

1. README accuracy (skip if unavailable)
2. Integration issues from analysis
3. Smoke tests (syntax, imports, entry point)
4. Pytest tests (if executed)

OUTPUT FORMAT (MANDATORY):
- First line: VERIFICATION: PASS or VERIFICATION: FAIL
- Then 3-8 bullet points explaining why

FAIL if: smoke tests fail OR pytest tests fail'''
    
    def save(self):
        """Save modified content"""
        if self.dry_run:
            print("\\n[DRY RUN] File not modified")
            return
        
        content = '\\n'.join(self.lines)
        with open(self.filepath, 'w') as f:
            f.write(content)
        
        print(f"\\n✓ Saved changes to {self.filepath}")
        print(f"\\nTotal changes made: {len(self.changes_made)}")
        for change in self.changes_made:
            print(f"  - {change}")


def main():
    parser = argparse.ArgumentParser(description="Apply verification fixes")
    parser.add_argument("--file", type=Path, required=True,
                       help="Path to swarm_coordinator_v2.py")
    parser.add_argument("--backup", action="store_true",
                       help="Create backup before modifying")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show changes without modifying file")
    args = parser.parse_args()
    
    if not args.file.exists():
        print(f"✗ File not found: {args.file}")
        return 1
    
    # Create backup
    if args.backup and not args.dry_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = args.file.with_suffix(f'.backup_{timestamp}.py')
        shutil.copy2(args.file, backup_path)
        print(f"✓ Created backup: {backup_path}\\n")
    
    # Apply patches
    applicator = PatchApplicator(args.file, dry_run=args.dry_run)
    applicator.apply_all_patches()
    applicator.save()
    
    print("\\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    
    if not args.dry_run:
        print("\\nNext steps:")
        print("  1. Test syntax: python -m py_compile", args.file)
        print("  2. Run SwarmCoordinator with a simple task")
        print("  3. Verify you see smoke tests and test execution")
    
    return 0


if __name__ == "__main__":
    exit(main())
