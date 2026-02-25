# Swarm Coordinator v2 Execution Flow

This document explains how the swarm coordinator executes workflows in swarm_coordinator_v2.py.

## Main Execution Loop

The execution loop is contained in the `run_workflow()` method (lines 4659-4780). The loop continues until all tasks are completed or max iterations are reached:

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
The `execute_tasks_parallel()` method (lines 3900-3920) handles parallel task execution using ThreadPoolExecutor.

### Individual Task Execution
The `execute_task()` method (lines 2284-3121) handles each individual task based on its assigned role:

1. **CLARIFIER**: Handles clarification tasks
2. **ARCHITECT**: Creates draft plans with validation retries
3. **CODER**: Generates code
4. **REVIEWER**: Reviews code
5. **DOCUMENTER**: Creates documentation
6. **SECURITY**: Performs security analysis
7. **VERIFIER**: Final verification

## Key Features

1. **Dependency Management**: Tasks execute only when dependencies are satisfied
2. **Revision Handling**: Automatic revision cycles when review tasks fail
3. **Interactive Clarification**: User interaction for clarification tasks
4. **Parallel Execution**: Tasks can run in parallel for efficiency
5. **Error Handling**: Comprehensive error handling with retries
6. **State Management**: Maintains state throughout workflow execution

## Workflow Types

The system supports multiple workflow types:
- standard, full, review_only, custom, planned, import, bugfix, collaborative

## Execution Cycle

1. **Task Creation**: Tasks created based on workflow type
2. **Task Execution**: Ready tasks executed in parallel
3. **State Update**: Completed tasks moved from queue to completed list
4. **Dependency Check**: New ready tasks identified for next iteration
5. **Revision Handling**: Revision cycles triggered when reviews fail
6. **Loop Continuation**: Continue until all tasks complete or max iterations reached
7. **Finalization**: Generate reports and save outputs