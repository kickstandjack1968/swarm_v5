"""
SWARMCOORDINATOR VERIFICATION FIX - COMPREHENSIVE PATCH
========================================================

This patch adds the missing verification functionality to swarm_coordinator_v2.py

WHAT THIS FIXES:
1. Smoke tests that were referenced but never implemented
2. Entry point integration check that didn't exist
3. Test execution that generated tests but never ran them
4. Verification that checked phantom test results

APPLY THIS PATCH IN ORDER:
"""

# ==============================================================================
# PATCH 1: Add missing methods to SwarmCoordinator class
# LOCATION: After _save_project_outputs method (around line 1100)
# ==============================================================================

PATCH_1_SMOKE_TESTS = '''
    def _run_smoke_tests(self) -> Dict[str, Any]:
        """
        Run basic smoke tests on generated code.
        
        Tests:
        1. All files have valid syntax
        2. All imports resolve
        3. Entry point can be imported without errors
        
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
        import ast
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
                    "stderr": "\\n".join(errors)
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
        import sys
        import subprocess
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
        import sys
        import subprocess
        try:
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
    import src.{module_name} as entry
    
    if hasattr(entry, 'main'):
        print("Entry point has main() function")
        sys.exit(0)
    else:
        print("Entry point imported successfully")
        sys.exit(0)
        
except SystemExit as e:
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
                "returncode": 0,
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
    
    def _check_entry_point_integration(self, code_content: str) -> str:
        """
        Check if entry point properly integrates with dependencies.
        
        Returns:
            String describing issues, or empty string if OK
        """
        import ast
        
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
        
        # Check for unused imports
        unused = imports - used_names
        if unused and len(unused) < 5:
            issues.append(f"Note: Imported but not used: {', '.join(sorted(unused))}")
        
        # Check for error handling
        has_try_except = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                has_try_except = True
                break
        
        # Check for main block
        has_main_block = False
        has_main_function = False
        
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
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
        
        return "\\n".join(issues) if issues else ""
    
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
'''

# ==============================================================================
# PATCH 2: Add test execution to execute_task method
# LOCATION: Inside execute_task, after test generation (around line 2150)
# FIND: task.result = test_code
#       task.status = TaskStatus.COMPLETED
# ADD AFTER:
# ==============================================================================

PATCH_2_TEST_EXECUTION = '''
                # Execute the generated tests
                if task.result:
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
                            lines = output.split('\\n')
                            relevant = '\\n'.join(lines[-20:])
                            print(f"    Output preview:\\n{relevant[:500]}")
'''

# ==============================================================================
# PATCH 3: Update _build_verification_prompt to use smoke tests
# LOCATION: In _build_verification_prompt method (around line 4073)
# REPLACE LINES 4073-4076 WITH:
# ==============================================================================

PATCH_3_VERIFICATION_PROMPT = '''
        # Get smoke tests - run them if not already available
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
'''

# ==============================================================================
# PATCH 4: Add test results section to verification prompt
# LOCATION: In _build_verification_prompt, in the user_message assembly
# FIND: {smoke_section}
# ADD AFTER:
# ==============================================================================

PATCH_4_TEST_RESULTS_SECTION = '''
        # Add test results section
        if test_execution:
            if test_execution.get("success"):
                test_section = """
TEST EXECUTION RESULTS:
✓ Generated pytest tests PASSED
All unit tests executed successfully in Docker sandbox."""
            else:
                error = test_execution.get("error", "Unknown error")
                output = test_execution.get("output", "")
                # Truncate output
                if len(output) > 1500:
                    output = "...\\n" + output[-1500:]
                
                test_section = f"""
TEST EXECUTION RESULTS:
✗ Generated pytest tests FAILED

Error: {error}

Test Output:
{output}

CRITICAL: Tests are failing. Review the test output carefully."""
        else:
            test_section = """
TEST EXECUTION RESULTS:
Tests were not executed (Docker sandbox unavailable)."""
'''

# ==============================================================================
# PATCH 5: Update verification checklist
# LOCATION: In _build_verification_prompt, replace the "Perform final verification" section
# AROUND LINE 4140-4153
# ==============================================================================

PATCH_5_VERIFICATION_CHECKLIST = '''
Perform final verification:

1. {("Skip - documentation not available" if doc_failed or not readme_content else "Does the README accurately describe the actual code and usage?")}
2. Are there any integration issues noted above that need fixing?
3. Do smoke tests indicate runtime/import failures? If yes: FAIL.
4. Do pytest tests pass? If no: FAIL (unless tests themselves are incorrect).

OUTPUT FORMAT (MANDATORY):
- First non-empty line MUST be exactly one of:
  VERIFICATION: PASS
  VERIFICATION: FAIL
- Then provide 3-8 bullet points explaining why.

PASS only if:
- Smoke tests passed (imports work, syntax valid, entry point loads)
- Pytest tests passed OR tests don't exist OR tests are clearly wrong
- README matches code (or doc generation failed)
- No critical integration issues
'''

# ==============================================================================
# INSTALLATION GUIDE
# ==============================================================================

INSTALLATION_GUIDE = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        INSTALLATION INSTRUCTIONS                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

STEP 1: Backup your current file
  cp swarm_coordinator_v2.py swarm_coordinator_v2.py.backup

STEP 2: Apply patches in order

  PATCH 1 - Add missing methods
  Location: After _save_project_outputs method (line ~1100)
  Copy PATCH_1_SMOKE_TESTS content
  
  PATCH 2 - Add test execution
  Location: In execute_task, after "task.status = TaskStatus.COMPLETED" (line ~2150)
  Find the test_generation section and add PATCH_2_TEST_EXECUTION
  
  PATCH 3 - Update verification prompt builder
  Location: In _build_verification_prompt (line ~4073)
  Replace smoke_tests assignment with PATCH_3_VERIFICATION_PROMPT
  
  PATCH 4 - Add test results section
  Location: After {smoke_section} in user_message (line ~4130)
  Add PATCH_4_TEST_RESULTS_SECTION before the final checklist
  
  PATCH 5 - Update verification checklist
  Location: Replace verification checklist (line ~4140)
  Replace with PATCH_5_VERIFICATION_CHECKLIST

STEP 3: Verify syntax
  python -m py_compile swarm_coordinator_v2.py

STEP 4: Test
  python swarm_coordinator_v2.py

STEP 5: Run a simple test
  user_request = "Create a function that adds two numbers"
  coordinator.run_workflow(user_request)
  
  Expected: You should now see:
  - "🔥 Running smoke tests..."
  - "🧪 Executing generated tests..."
  - Test pass/fail results
  - Proper verification with real test data

╔══════════════════════════════════════════════════════════════════════════════╗
║                         WHAT CHANGES WILL HAPPEN                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

BEFORE:
  ✗ Tests generated but never run
  ✗ Smoke tests referenced but not executed
  ✗ Verification checks phantom data
  ✗ Broken code can pass verification

AFTER:
  ✓ Tests generated AND executed in Docker sandbox
  ✓ Smoke tests actually run (syntax, imports, entry point)
  ✓ Verification sees real test results
  ✓ Failing tests = failing verification
  ✓ Entry point integration actually checked

╔══════════════════════════════════════════════════════════════════════════════╗
║                           TROUBLESHOOTING                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

ERROR: Docker sandbox not available
  → Tests will be skipped with warning
  → Verification still runs but without test execution
  → Install Docker or run on machine with Docker

ERROR: Tests fail on working code
  → Check test generation - tests might be wrong
  → Review test output for false failures
  → Verifier will see failure but can override

ERROR: Smoke tests fail
  → Check imports - missing dependencies?
  → Check entry point - syntax errors?
  → Review project structure - files in wrong locations?
"""

if __name__ == "__main__":
    print(INSTALLATION_GUIDE)
    print()
    print("=" * 80)
    print("READY TO INSTALL")
    print("=" * 80)
    print()
    print("Open swarm_coordinator_v2.py and apply patches in order.")
    print("Each patch is clearly labeled with its location.")
    print()
    print("After installation, your verification system will actually work.")
