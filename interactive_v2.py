#!/usr/bin/env python3
"""
Interactive CLI for Advanced Swarm Coordinator v2 - WITH PROJECT IMPORT
Enhanced interface with workflow selection and real-time monitoring
"""

import sys
import os
import json

# Add current directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from swarm_coordinator_v2 import SwarmCoordinator, Task, AgentRole, TaskStatus


def print_banner():
    """Print welcome banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                   ADVANCED MULTI-AGENT SWARM v2.0 (FIXED)                    ║
║                                                                              ║
║  Parallel Execution • Dynamic Routing • Enhanced Observability               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)
    
    print("\n🤖 Available Agent Roles:")
    print("   • ARCHITECT    - System design and architecture")
    print("   • CLARIFIER    - Requirements clarification")
    print("   • CODER        - Code implementation")
    print("   • REVIEWER     - Code review and quality")
    print("   • TESTER       - Test generation")
    print("   • OPTIMIZER    - Performance optimization")
    print("   • DOCUMENTER   - Documentation generation")
    print("   • DEBUGGER     - Bug analysis and fixes")
    print("   • SECURITY     - Security analysis")
    print("   • VERIFIER     - Final verification")
    print("\n" + "=" * 80 + "\n")


def get_multiline_input(prompt: str) -> str:
    """Get multiline input from user"""
    print(prompt)
    print("(Type your input. Enter 'END' on a new line when done)")
    print("-" * 80)
    
    lines = []
    while True:
        try:
            line = input()
            if line.strip().upper() == 'END':
                break
            lines.append(line)
        except EOFError:
            break
    
    return '\n'.join(lines)


def select_workflow() -> str:
    """Let user select workflow type"""
    print("\n📋 Select Workflow Type:")
    print("   1. STANDARD  - Clarify → Architect → Code → Review → Document → Verify")
    print("   2. FULL      - Complete pipeline with all agents (Comprehensive)")
    print("   3. REVIEW    - Review existing code (4 parallel reviews)")
    print("   4. CUSTOM    - Build custom workflow")
    print("   5. PLANNED   - File-by-file execution with YAML plan")
    print("   6. IMPORT    - Import existing project and modify it")
    print("   7. BUGFIX    - Debug and fix bugs in existing code")
    print("   8. COLLABORATIVE - Clarify → Draft Plan → Coder Reviews → Final Plan → Build → Compliance Review")

    while True:
        choice = input("\nYour choice [1-8]: ").strip()
        if choice == '1':
            return 'standard'
        elif choice == '2':
            return 'full'
        elif choice == '3':
            return 'review_only'
        elif choice == '4':
            return 'custom'
        elif choice == '5':
            return 'planned'
        elif choice == '6':
            return 'import'
        elif choice == '7':
            return 'bugfix'
        elif choice == '8':
            return 'collaborative'
        else:
            print("❌ Invalid choice. Please enter 1-8.")


def select_stop_after() -> str:
    """Let user select which collaborative stage to stop after."""
    print("\n⏸ Stop after which stage?")
    print("   1. CLARIFY        - Stop after clarifier produces job scope")
    print("   2. DRAFT_PLAN     - Stop after architect drafts the plan")
    print("   3. PLAN_REVIEW    - Stop after coder reviews the plan")
    print("   4. FINALIZE_PLAN  - Stop after architect finalizes the plan")
    print("   5. BUILD          - Stop after coder builds all files")
    print("   6. COMPLIANCE     - Run everything (full pipeline)")

    stage_map = {
        '1': 'clarify',
        '2': 'draft_plan',
        '3': 'plan_review',
        '4': 'finalize_plan',
        '5': 'build',
        '6': None,  # None = run all
    }

    while True:
        choice = input("\nYour choice [1-6]: ").strip()
        if choice in stage_map:
            return stage_map[choice]
        print("❌ Invalid choice. Please enter 1-6.")


def configure_custom_workflow(coordinator: SwarmCoordinator) -> str:
    """Allow user to build custom workflow and return the project description"""
    print("\n🔧 Custom Workflow Builder")
    print("=" * 80)
    
    tasks = []
    task_counter = 1
    
    print("\nAvailable agent roles:")
    roles = list(AgentRole)
    for i, role in enumerate(roles, 1):
        print(f"   {i:2d}. {role.value}")
    
    while True:
        print(f"\n--- Task #{task_counter} ---")
        
        role_input = input("Select agent role (1-10) or 'done' to finish: ").strip()
        if role_input.lower() == 'done':
            break
        
        try:
            role_idx = int(role_input) - 1
            if role_idx < 0 or role_idx >= len(roles):
                raise ValueError("Index out of range")
            role = roles[role_idx]
        except (ValueError, IndexError):
            print("❌ Invalid role. Try again.")
            continue
        
        description = input(f"Task description for {role.value}: ").strip()
        if not description:
            description = f"Task for {role.value}"
        
        priority = input("Priority (1-10, default 5): ").strip()
        priority = int(priority) if priority.isdigit() and 1 <= int(priority) <= 10 else 5
        
        if tasks:
            print(f"\nExisting tasks: {', '.join([t['task_id'] for t in tasks])}")
            deps = input("Dependencies (comma-separated task IDs, or empty): ").strip()
            dependencies = [d.strip() for d in deps.split(',') if d.strip()] if deps else []
        else:
            dependencies = []
        
        task_id = f"T{task_counter:03d}_{role.value}"
        tasks.append({
            'task_id': task_id,
            'role': role,
            'description': description,
            'priority': priority,
            'dependencies': dependencies
        })
        
        print(f"✓ Added task: {task_id}")
        task_counter += 1
    
    if not tasks:
        print("⚠ No tasks added to custom workflow")
        return ""
    
    # Get overall project description
    user_request = get_multiline_input("\nEnter the overall project description:")
    
    # Add tasks to coordinator
    for task_info in tasks:
        coordinator.add_task(Task(
            task_id=task_info['task_id'],
            task_type=task_info['role'].value,
            description=task_info['description'],
            assigned_role=task_info['role'],
            status=TaskStatus.PENDING,
            priority=task_info['priority'],
            dependencies=task_info['dependencies'],
            metadata={"user_request": user_request}
        ))
    
    print(f"\n✓ Custom workflow created with {len(tasks)} tasks")
    return user_request


def handle_import_workflow(coordinator: SwarmCoordinator) -> str:
    """Handle the import workflow - import existing project and set up tasks."""
    try:
        from project_import import (
            import_existing_project, 
            create_import_workflow, 
            get_import_prompt_interactive,
            setup_import_project_directory
        )
    except ImportError as e:
        print(f"\n❌ Import workflow not available: {e}")
        print("   Ensure project_import.py and related modules are in the src/ directory")
        return ""
    
    # Get project path and task description from user
    project_path, task_description = get_import_prompt_interactive()
    
    if not task_description.strip():
        print("\n❌ No task description provided. Exiting.")
        return ""
    
    try:
        # Import the project
        imported_project = import_existing_project(project_path)
        
        # --- NEW CODE: Actually copy the files ---
        print(f"\n📁 Setting up project directory...")
        setup_import_project_directory(coordinator, project_path, task_description)

        # Create workflow tasks
        create_import_workflow(coordinator, imported_project, task_description)
        
        # Store workspace info for later
        coordinator.state["project_info"]["source_project"] = project_path
        coordinator.state["project_info"]["workspace_dir"] = imported_project.get("path", project_path)
        
        return task_description
        
    except Exception as e:
        print(f"\n❌ Failed to import project: {e}")
        import traceback
        traceback.print_exc()
        return ""


def handle_bugfix_workflow(coordinator: SwarmCoordinator) -> str:
    """Handle the bugfix workflow - collect bug info and set up debugging tasks."""
    try:
        from bugfix_workflow import get_bugfix_input, create_bugfix_workflow
    except ImportError as e:
        print(f"\n❌ Bugfix workflow not available: {e}")
        print("   Ensure bugfix_workflow.py is in the src/ directory")
        return ""
    
    try:
        # Get bug details from user
        bugfix_context = get_bugfix_input()
        
        # Create workflow tasks
        create_bugfix_workflow(coordinator, bugfix_context)
        
        return f"Fix bug: {bugfix_context.expected_behavior}"
        
    except Exception as e:
        print(f"\n❌ Failed to set up bugfix workflow: {e}")
        import traceback
        traceback.print_exc()
        return ""


def check_configuration() -> bool:
    """Check and display configuration"""
    config_paths = ["config_v2.json", "config/config_v2.json"]
    config_file = None
    
    for path in config_paths:
        if os.path.exists(path):
            config_file = path
            break
    
    if not config_file:
        print(f"⚠ Warning: config_v2.json not found")
        print("Will use default configuration...")
        return True
    
    print(f"✓ Found {config_file}")
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"⚠ Warning: Invalid JSON in config file: {e}")
        print("Will use default configuration...")
        return True
    
    mode = config.get('model_config', {}).get('mode', 'single')
    print(f"   Mode: {mode}")
    
    if mode == 'multi':
        print("   Multi-model configuration:")
        multi_config = config.get('model_config', {}).get('multi_model', {})
        for agent, info in multi_config.items():
            model_name = info.get('model_name', info.get('model', 'Unknown'))
            api_type = info.get('api_type', 'Unknown')
            print(f"      • {agent:12s}: {model_name[:40]:40s} ({api_type})")
    else:
        single_config = config.get('model_config', {}).get('single_model', {})
        print(f"   Single model: {single_config.get('model', 'Unknown')}")
        print(f"   URL: {single_config.get('url', 'Unknown')}")
    
    workflow = config.get('workflow', {})
    parallel_enabled = workflow.get('enable_parallel', False)
    max_parallel = workflow.get('max_parallel_agents', 4)
    max_iterations = workflow.get('max_iterations', 3)
    
    print(f"   Parallel execution: {'Enabled' if parallel_enabled else 'Disabled'}")
    if parallel_enabled:
        print(f"   Max parallel agents: {max_parallel}")
    print(f"   Max iterations: {max_iterations}")
    
    use_config = input("\nUse this configuration? [Y/n]: ").strip().lower()
    return not use_config or use_config == 'y'


def save_outputs(coordinator: SwarmCoordinator):
    """Save outputs to files"""
    print("\n💾 Saving outputs...")
    
    project_dir = coordinator.state.get("project_info", {}).get("project_dir")
    
    if project_dir and os.path.exists(project_dir):
        print(f"   ✓ All files saved to project directory:")
        print(f"      {project_dir}")
        
        # List what was saved
        for subdir in ["src", "tests", "docs"]:
            subdir_path = os.path.join(project_dir, subdir)
            if os.path.exists(subdir_path):
                files = os.listdir(subdir_path)
                if files:
                    print(f"   ✓ {subdir.title()}: {', '.join(files)}")
        
        # Root files
        root_files = [f for f in os.listdir(project_dir) 
                     if os.path.isfile(os.path.join(project_dir, f))]
        if root_files:
            print(f"   ✓ Root files: {', '.join(root_files)}")
    else:
        print("   ⚠ No project directory found")
        
        # Fallback: save code to current directory
        code_task = next((t for t in coordinator.completed_tasks 
                         if t.task_type == "coding" and t.result), None)
        if code_task:
            with open("generated_code.py", 'w') as f:
                f.write(code_task.result)
            print("   ✓ Code saved to: generated_code.py")


def main():
    """Main entry point"""
    try:
        print_banner()
        
        # Check configuration
        if not check_configuration():
            print("\n❌ Configuration rejected. Exiting.")
            return
        
        # Create coordinator
        print("\n🚀 Initializing swarm coordinator...")
        
        # Find config file
        config_file = "config_v2.json"
        if not os.path.exists(config_file) and os.path.exists("config/config_v2.json"):
            config_file = "config/config_v2.json"
        
        coordinator = SwarmCoordinator(config_file=config_file)
        print("✓ Coordinator ready")
        
        # Select workflow
        workflow_type = select_workflow()
        
        if workflow_type == 'custom':
            user_request = configure_custom_workflow(coordinator)
            
            if not user_request and not coordinator.task_queue:
                print("\n❌ No tasks or description provided. Exiting.")
                return
            
            print("\n" + "=" * 80)
            print("EXECUTING CUSTOM WORKFLOW")
            print("=" * 80)
            
            coordinator.run_workflow(user_request, workflow_type="custom")
            
        elif workflow_type == 'review_only':
            code = get_multiline_input("\nEnter the code to review:")
            
            if not code.strip():
                print("\n❌ No code provided. Exiting.")
                return
            
            print("\n" + "=" * 80)
            print("EXECUTING REVIEW WORKFLOW")
            print("=" * 80)
            
            coordinator.run_workflow(code, workflow_type="review_only")
            
        elif workflow_type == 'import':
            # Handle import workflow
            user_request = handle_import_workflow(coordinator)
            
            if not user_request:
                return
            
            print("\n" + "=" * 80)
            print("EXECUTING IMPORT WORKFLOW")
            print("=" * 80)
            
            # Run with the tasks already added by handle_import_workflow
            # Use "import" workflow type so coordinator knows to use import-specific handling
            coordinator.run_workflow(user_request, workflow_type="import")
            
        elif workflow_type == 'bugfix':
            # Handle bugfix workflow
            user_request = handle_bugfix_workflow(coordinator)
            
            if not user_request:
                return
            
            print("\n" + "=" * 80)
            print("EXECUTING BUGFIX WORKFLOW")
            print("=" * 80)
            
            # Run with the tasks already added by handle_bugfix_workflow
            coordinator.run_workflow(user_request, workflow_type="bugfix")
            
        else:
            user_request = get_multiline_input(
                f"\n{'='*80}\nWhat would you like to build?\n{'='*80}\n"
            )

            if not user_request.strip():
                print("\n❌ No request provided. Exiting.")
                return

            # For collaborative workflow, ask which stage to stop after
            stop_after = None
            if workflow_type == 'collaborative':
                stop_after = select_stop_after()

            print("\n" + "=" * 80)
            print(f"EXECUTING {workflow_type.upper()} WORKFLOW")
            print("=" * 80)

            coordinator.run_workflow(user_request, workflow_type=workflow_type, stop_after=stop_after)
        
        # Save outputs
        save_outputs(coordinator)
        
        # Display metrics summary
        print("\n" + "=" * 80)
        print("METRICS SUMMARY")
        print("=" * 80)
        metrics = coordinator.get_metrics_summary()
        print(json.dumps(metrics, indent=2))
        
        print("\n✓ Workflow complete!")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Workflow interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
