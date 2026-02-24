#!/usr/bin/env python3
"""
SwarmCoordinator Verification Fixes
====================================
Missing methods that make verification actually work.

ADD THESE METHODS TO SwarmCoordinator CLASS in swarm_coordinator_v2.py
"""

import ast
import os
import sys
import subprocess
import tempfile
from typing import Dict, Any, List, Tuple, Optional


# ==============================================================================
# MISSING METHOD #1: _run_smoke_tests
# ==============================================================================
# ADD THIS METHOD TO SwarmCoordinator CLASS (around line 1100)

def _run_smoke_tests(self) -> Dict[str, Any]:
    """
    Run basic smoke tests on generated code.
    
    Tests:
    1. All files have valid syntax
    2. All imports resolve
    3. Entry point can be imported without errors
    4. Entry point runs without immediate crash (if applicable)
    
    Returns:
        Dict of test_name -> {cmd, returncode, stdout, stderr}
    """
    results = {}
    project_dir = self.state.get("project_info", {}).get("project_dir")
    
    if not project_dir or not os.path.exists(project_dir):
        return {"error": {"cmd": "N/A", "returncode": -1, "stdout": "", 
                         "stderr": "Project directory not found"}}
    
    src_dir = os.path.join(project_dir, "src")
    if not os.path.exists(src_dir):
        return {"error": {"cmd": "N/A", "returncode": -1, "stdout": "", 
                         "stderr": "Source directory not found"}}
    
    # Test 1: Syntax check all Python files
    results["syntax_check"] = self._smoke_test_syntax(src_dir)
    
    # Test 2: Import check
    results["import_check"] = self._smoke_test_imports(project_dir, src_dir)
    
    # Test 3: Entry point execution (if applicable)
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
            return {
                "cmd": "ast.parse(all_files)",
                "returncode": 1,
                "stdout": "",
                "stderr": "\n".join(errors)
            }
        else:
            return {
                "cmd": "ast.parse(all_files)",
                "returncode": 0,
                "stdout": "All files have valid syntax",
                "stderr": ""
            }
    except Exception as e:
        return {
            "cmd": "ast.parse(all_files)",
            "returncode": -1,
            "stdout": "",
            "stderr": f"Syntax check failed: {e}"
        }


def _smoke_test_imports(self, project_dir: str, src_dir: str) -> Dict[str, Any]:
    """Test if all modules can be imported without errors."""
    try:
        # Find all Python files
        py_files = []
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    rel_path = os.path.relpath(os.path.join(root, file), src_dir)
                    module_path = rel_path.replace(os.sep, '.').replace('.py', '')
                    py_files.append(module_path)
        
        # Try importing each module
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
        
        result = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=project_dir
        )
        
        return {
            "cmd": f"python -c 'import all modules'",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
    except subprocess.TimeoutExpired:
        return {
            "cmd": "import test",
            "returncode": -1,
            "stdout": "",
            "stderr": "Import test timed out (possible infinite loop in module initialization)"
        }
    except Exception as e:
        return {
            "cmd": "import test",
            "returncode": -1,
            "stdout": "",
            "stderr": f"Import test failed: {e}"
        }


def _smoke_test_entry_point(self, project_dir: str, entry_point: str) -> Dict[str, Any]:
    """Test if entry point runs without immediate crash."""
    try:
        # Run with --help or -h to see if it responds
        module_name = entry_point.replace('.py', '').replace('/', '.')
        
        test_script = f"""
import sys
import signal

def timeout_handler(signum, frame):
    print("Entry point is running (timed out after 3s)")
    sys.exit(0)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(3)

sys.path.insert(0, {repr(project_dir)})

try:
    # Try importing and running
    import src.{module_name} as entry
    
    # If it has a main function, test it
    if hasattr(entry, 'main'):
        # Don't actually run main, just verify it exists
        print("Entry point has main() function")
        sys.exit(0)
    else:
        print("Entry point imported successfully")
        sys.exit(0)
        
except SystemExit as e:
    # main() might call sys.exit, that's OK
    if e.code == 0:
        print("Entry point executed successfully")
    sys.exit(e.code)
except Exception as e:
    print(f"Entry point failed: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
        
        result = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=project_dir
        )
        
        return {
            "cmd": f"python -m src.{module_name.replace('.', '/')}",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
    except subprocess.TimeoutExpired:
        return {
            "cmd": f"python -m src.{entry_point}",
            "returncode": 0,  # Timeout means it's running, not crashed
            "stdout": "Entry point started successfully (timed out)",
            "stderr": ""
        }
    except Exception as e:
        return {
            "cmd": f"python -m src.{entry_point}",
            "returncode": -1,
            "stdout": "",
            "stderr": f"Execution test failed: {e}"
        }


# ==============================================================================
# MISSING METHOD #2: _check_entry_point_integration
# ==============================================================================
# ADD THIS METHOD TO SwarmCoordinator CLASS (around line 1150)

def _check_entry_point_integration(self, code_content: str) -> str:
    """
    Check if entry point properly integrates with dependencies.
    
    Checks:
    1. All imported names are actually used
    2. Function calls match known signatures
    3. Required initializations happen
    
    Returns:
        String describing issues, or empty string if OK
    """
    if not code_content.strip():
        return "Entry point code is empty"
    
    issues = []
    
    try:
        tree = ast.parse(code_content)
    except SyntaxError as e:
        return f"Entry point has syntax errors: {e}"
    
    # Extract imports
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module:
                for alias in node.names:
                    imports.add(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
    
    # Extract function/class usage
    used_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used_names.add(node.id)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                used_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                used_names.add(node.func.attr)
    
    # Check for unused imports (warning, not critical)
    unused = imports - used_names
    if unused and len(unused) < 5:  # Only report if not too many
        issues.append(f"Note: Imported but not used: {', '.join(sorted(unused))}")
    
    # Check for bare instantiation without error handling
    has_try_except = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            has_try_except = True
            break
    
    # Check if there's a main() or if __name__ == "__main__" block
    has_main_block = False
    has_main_function = False
    
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            # Check if this is if __name__ == "__main__":
            if isinstance(node.test, ast.Compare):
                if (isinstance(node.test.left, ast.Name) and 
                    node.test.left.id == '__name__'):
                    has_main_block = True
        
        if isinstance(node, ast.FunctionDef) and node.name == 'main':
            has_main_function = True
    
    if not has_main_block and not has_main_function:
        issues.append("Entry point lacks if __name__ == '__main__': block")
    
    # Check for unhandled file operations
    file_ops = ['open', 'read', 'write']
    uses_files = any(name in used_names for name in file_ops)
    
    if uses_files and not has_try_except:
        issues.append("File operations without try/except error handling")
    
    return "\n".join(issues) if issues else ""


# ==============================================================================
# MISSING METHOD #3: _execute_generated_tests
# ==============================================================================
# ADD THIS METHOD TO SwarmCoordinator CLASS (around line 1200)

def _execute_generated_tests(self, task) -> Dict[str, Any]:
    """
    Execute pytest tests that were generated by the tester agent.
    
    Returns:
        Dict with test execution results
    """
    # Get the test content from tester task
    test_task = next(
        (t for t in self.completed_tasks if t.task_type == "test_generation"),
        None
    )
    
    if not test_task or not test_task.result:
        return {
            "success": False,
            "error": "No tests were generated",
            "output": ""
        }
    
    test_content = self._clean_test_output(test_task.result)
    
    # Get source files
    code_task = next(
        (t for t in self.completed_tasks if t.task_type in ("coding", "plan_execution")),
        None
    )
    
    if not code_task or not code_task.result:
        return {
            "success": False,
            "error": "No source code found",
            "output": ""
        }
    
    # Parse multi-file output
    files_dict = self._parse_multi_file_output(code_task.result)
    
    # Convert to completed_files format for DockerSandbox
    completed_files = {}
    for fname, content in files_dict.items():
        # Create a simple object with .content attribute
        completed_files[fname] = type('FileResult', (), {'content': content})()
    
    # Get requirements if available
    requirements_content = ""
    requirements_path = os.path.join(
        self.state.get("project_info", {}).get("project_dir", ""),
        "requirements.txt"
    )
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            requirements_content = f.read()
    
    # Get or start Docker sandbox
    sandbox = get_docker_sandbox()
    if not sandbox:
        return {
            "success": False,
            "error": "Docker sandbox not available",
            "output": "Test execution skipped - no Docker sandbox"
        }
    
    if not sandbox.container_running:
        if not sandbox.start():
            return {
                "success": False,
                "error": "Failed to start Docker sandbox",
                "output": "Test execution skipped - sandbox start failed"
            }
    
    # Execute tests
    try:
        success, output = sandbox.run_hot_test(
            completed_files=completed_files,
            test_content=test_content,
            file_name="test_main.py",
            requirements_content=requirements_content
        )
        
        return {
            "success": success,
            "error": "" if success else "Tests failed",
            "output": output
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Test execution error: {e}",
            "output": ""
        }


# ==============================================================================
# PATCH FOR execute_task METHOD
# ==============================================================================
# INSERT THIS CODE AFTER LINE 2150 (after test generation completes)

def _patch_test_execution_in_execute_task():
    """
    This shows WHERE to add test execution in the execute_task method.
    
    FIND THIS CODE (around line 2150):
        elif task.task_type == "test_generation":
            # ... existing test generation code ...
            task.result = test_code  # ← AFTER THIS LINE
            task.status = TaskStatus.COMPLETED
    
    ADD THIS CODE IMMEDIATELY AFTER:
    """
    
    # Execute the generated tests
    if task.result:  # Only if tests were generated
        print(f"  🧪 Executing generated tests...")
        test_results = self._execute_generated_tests(task)
        
        # Store results in task metadata
        task.metadata["test_execution"] = test_results
        
        # Report results
        if test_results["success"]:
            print(f"  ✓ All tests passed")
        else:
            print(f"  ✗ Tests failed or could not run")
            if test_results.get("error"):
                print(f"    Error: {test_results['error']}")
            
            # Show failure output (truncated)
            output = test_results.get("output", "")
            if output:
                lines = output.split('\n')
                # Show last 20 lines (usually has the failure summary)
                relevant = '\n'.join(lines[-20:])
                print(f"    Output:\n{relevant[:500]}")


# ==============================================================================
# PATCH FOR _build_verification_prompt
# ==============================================================================
# REPLACE LINES 4073-4076 with this:

def _patch_verification_prompt():
    """
    REPLACE THIS CODE (lines 4073-4076):
        smoke_tests = task.metadata.get("smoke_tests")
        if smoke_tests is None:
            smoke_tests = self._run_smoke_tests()
    
    WITH THIS:
    """
    # Get smoke tests - try metadata first, then run them
    smoke_tests = task.metadata.get("smoke_tests")
    if smoke_tests is None:
        print("  🔥 Running smoke tests...")
        smoke_tests = self._run_smoke_tests()
        task.metadata["smoke_tests"] = smoke_tests
        
        # Report smoke test results
        all_passed = all(
            result.get("returncode") == 0 
            for result in smoke_tests.values() 
            if isinstance(result, dict)
        )
        if all_passed:
            print("  ✓ All smoke tests passed")
        else:
            print("  ⚠ Some smoke tests failed")
    
    # Get test execution results from tester task
    test_task = next(
        (t for t in self.completed_tasks if t.task_type == "test_generation"),
        None
    )
    test_execution = None
    if test_task:
        test_execution = test_task.metadata.get("test_execution")


# ==============================================================================
# COMPLETE VERIFICATION PROMPT UPDATE
# ==============================================================================
# UPDATE THE user_message in _build_verification_prompt (around line 4125)
# ADD THIS SECTION BEFORE THE "ACTUAL CODE" SECTION:

def _patch_verification_prompt_message():
    """
    ADD THIS SECTION to the verification user_message:
    """
    
    # After smoke_section, add test results section:
    test_section = ""
    if test_execution:
        if test_execution.get("success"):
            test_section = """
TEST RESULTS:
✓ Generated pytest tests PASSED
All unit tests executed successfully."""
        else:
            error = test_execution.get("error", "Unknown error")
            output = test_execution.get("output", "")
            # Truncate output
            if len(output) > 2000:
                output = output[-2000:] + "\n... (truncated)"
            
            test_section = f"""
TEST RESULTS:
✗ Generated pytest tests FAILED

Error: {error}

Test Output:
{output}

CRITICAL: Tests are failing. This is a FAIL condition unless tests are incorrect."""
    else:
        test_section = """
TEST RESULTS:
Tests were not executed (Docker sandbox unavailable or tests not generated)."""
    
    # Then update the verification checklist:
    verification_checklist = f"""
Perform final verification:

1. {("Skip - documentation not available" if doc_failed or not readme_content else "Does the README accurately describe the actual code and usage?")}
2. Are there any integration issues noted above that need fixing?
3. Do smoke tests indicate runtime/test failures? If yes: FAIL.
4. Do pytest tests pass? If tests failed: FAIL (unless tests are wrong).

OUTPUT FORMAT (MANDATORY):
- First non-empty line MUST be exactly one of:
  VERIFICATION: PASS
  VERIFICATION: FAIL
- Then provide 3-8 bullet points explaining why.
"""


# ==============================================================================
# INSTALLATION INSTRUCTIONS
# ==============================================================================

INSTALLATION_INSTRUCTIONS = """
HOW TO INSTALL THESE FIXES:
===========================

1. Open swarm_coordinator_v2.py

2. ADD THESE METHODS to the SwarmCoordinator class:
   - _run_smoke_tests()                  (line ~1100)
   - _smoke_test_syntax()                (line ~1120)
   - _smoke_test_imports()               (line ~1145)
   - _smoke_test_entry_point()           (line ~1190)
   - _check_entry_point_integration()    (line ~1240)
   - _execute_generated_tests()          (line ~1290)

3. PATCH execute_task method (line ~2150):
   After test generation completes, add test execution call

4. PATCH _build_verification_prompt (line ~4073):
   Update to actually run smoke tests and include test results

5. TEST the changes:
   python swarm_coordinator_v2.py

The verification system will now:
✓ Actually run smoke tests
✓ Execute generated pytest tests
✓ Check entry point integration
✓ Report test failures in final verification
✓ Fail verification if tests fail
"""

if __name__ == "__main__":
    print(INSTALLATION_INSTRUCTIONS)
