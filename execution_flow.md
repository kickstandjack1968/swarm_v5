# Execution Flow in swarm_coordinator_v2.py

## Overview
The swarm coordinator uses a task-based execution loop that runs workflows in iterative cycles until all tasks are completed.

## Main Execution Loop

### Entry Point
- `run_workflow()` method (lines 4659-4780) is the main entry point

### Loop Structure
The execution loop runs in `run_workflow()` method (lines 4734-4763):

```python
while self.task_queue and iteration < max_iterations * 10:
    iteration += 1
    ready_tasks = self.get_ready_tasks()

    if not ready_tasks:
        # Handle blocked tasks
        break

    print(f"\n▶ Iteration {iteration}: Executing {len(ready_tasks)} tasks")

    completed = self.execute_tasks_parallel(ready_tasks)

    for task in completed:
        if task in self.task_queue:
            self.task_queue.remove(task)
        self.completed_tasks.append(task)

        # Handle task completion
        if task.task_type == "clarification" and task.status == TaskStatus.COMPLETED:
            self._handle_clarification_interactive(task)

        # Handle revision cycles
        review_tasks = [t for t in completed if "review" in t.task_type]
        if review_tasks:
            self._handle_revision_cycle(review_tasks)
```

## Task Execution Flow

### Parallel Execution
- `execute_tasks_parallel()` method (lines 3900-3920) handles parallel task execution
- Uses `ThreadPoolExecutor` for concurrent task execution
- Falls back to sequential execution if parallelism is disabled or only one task

### Individual Task Execution
- `execute_task()` method (lines 2284-3121) handles each individual task
- Task execution depends on the assigned role:
  - CLARIFIER: Clarification tasks
  - ARCHITECT: Draft plan creation with validation retries
  - CODER: Code generation
  - REVIEWER: Code review
  - DOCUMENTER: Documentation generation
  - SECURITY: Security analysis
  - VERIFIER: Final verification

## Workflow Types

The system supports multiple workflow types:
- standard, full, review_only, custom, planned, import, bugfix, collaborative

## Key Features

1. **Dependency Management**: Tasks are executed only when their dependencies are satisfied
2. **Revision Handling**: Automatic revision cycles when review tasks fail
3. **Interactive Clarification**: User interaction for clarification tasks
4. **Parallel Execution**: Tasks can run in parallel for efficiency
5. **Error Handling**: Comprehensive error handling with retries where appropriate
6. **State Management**: Maintains state throughout the workflow execution

## Execution Cycle

1. **Task Creation**: Based on workflow type, tasks are created and added to task queue
2. **Task Execution**: Ready tasks are executed in parallel
3. **State Update**: Completed tasks are moved from queue to completed list
4. **Dependency Check**: New ready tasks are identified for next iteration
5. **Revision Handling**: If review tasks fail, revision cycles are triggered
6. **Loop Continuation**: Continue until all tasks complete or max iterations reached
7. **Finalization**: Generate reports and save outputs