"""
Swarm UI Service - Bridge between Flask routes and SwarmCoordinatorUI.

Handles:
- Task creation and management
- Progress streaming to SSE
- Clarification flow management
- File handling
"""

import json
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import Queue
from threading import Thread

from .swarm_coordinator_ui import (
    SwarmCoordinatorUI,
    SwarmState,
    ProgressUpdate,
    create_swarm_ui
)


class UITaskStatus(Enum):
    """Status for UI-level task tracking"""
    PENDING = "pending"
    RUNNING = "running"
    WAITING_INPUT = "waiting_input"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class UITask:
    """UI-level task wrapper"""
    task_id: str
    description: str
    workflow_type: str
    files: List[str] = field(default_factory=list)
    status: UITaskStatus = UITaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    output_path: Optional[str] = None
    error: Optional[str] = None
    pending_question: Optional[Dict] = None
    coordinator: Optional[SwarmCoordinatorUI] = None


class SwarmUIService:
    """
    Service layer for UI swarm operations.
    
    Manages task lifecycle and bridges Flask routes to SwarmCoordinatorUI.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize service.
        
        Args:
            config: Optional configuration dict for SwarmCoordinatorUI
        """
        self.config = config or {}
        self.tasks: Dict[str, UITask] = {}
        self.progress_queues: Dict[str, Queue] = {}
    
    def create_task(
        self,
        description: str,
        workflow_type: str = "standard",
        files: Optional[List[str]] = None
    ) -> UITask:
        """
        Create a new swarm task.
        
        Args:
            description: Task description
            workflow_type: 'standard', 'data', 'quick'
            files: Optional list of input file paths
        
        Returns:
            Created UITask
        """
        task_id = str(uuid.uuid4())[:8]
        
        task = UITask(
            task_id=task_id,
            description=description,
            workflow_type=workflow_type,
            files=files or []
        )
        
        self.tasks[task_id] = task
        self.progress_queues[task_id] = Queue()
        
        return task
    
    def execute_task(self, task_id: str) -> Iterator[Dict[str, Any]]:
        """
        Execute a task and yield progress updates.
        
        Args:
            task_id: ID of task to execute
        
        Yields:
            Progress update dicts for SSE streaming
        """
        task = self.tasks.get(task_id)
        if not task:
            yield {'type': 'error', 'message': f'Task {task_id} not found'}
            return
        
        task.status = UITaskStatus.RUNNING
        task.started_at = datetime.now()
        
        yield {
            'type': 'status',
            'task_id': task_id,
            'status': 'started',
            'message': 'Swarm starting...'
        }
        
        try:
            # Create coordinator for this task
            coordinator = create_swarm_ui(self.config)
            task.coordinator = coordinator
            
            # Set up progress callback
            progress_queue = self.progress_queues[task_id]
            
            def progress_callback(update: ProgressUpdate):
                progress_queue.put(update.to_dict())
            
            coordinator.set_progress_callback(progress_callback)
            
            # Run workflow in background thread
            def run_workflow():
                try:
                    for update in coordinator.run_workflow(
                        user_request=task.description,
                        workflow_type=task.workflow_type,
                        files=task.files
                    ):
                        progress_queue.put(update.to_dict())
                except Exception as e:
                    progress_queue.put({
                        'type': 'error',
                        'message': str(e)
                    })
                finally:
                    progress_queue.put(None)  # Signal completion
            
            thread = Thread(target=run_workflow)
            thread.start()
            
            # Yield progress updates
            while True:
                update = progress_queue.get()
                
                if update is None:
                    break
                
                # Handle waiting for input
                if update.get('type') == 'question':
                    task.status = UITaskStatus.WAITING_INPUT
                    task.pending_question = update.get('data', {})
                
                # Handle completion
                if update.get('type') == 'complete':
                    task.status = UITaskStatus.COMPLETED
                    task.completed_at = datetime.now()
                    task.output_path = update.get('data', {}).get('project_dir')
                
                # Handle error
                if update.get('type') == 'error':
                    task.status = UITaskStatus.FAILED
                    task.error = update.get('message')
                
                yield update
            
            thread.join(timeout=5)
            
        except Exception as e:
            task.status = UITaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            
            yield {
                'type': 'error',
                'task_id': task_id,
                'message': str(e)
            }
    
    def provide_answer(self, task_id: str, answer: str) -> Iterator[Dict[str, Any]]:
        """
        Provide answer to pending clarification question.
        
        Args:
            task_id: Task ID
            answer: User's answer
        
        Yields:
            Progress updates as workflow continues
        """
        task = self.tasks.get(task_id)
        if not task:
            yield {'type': 'error', 'message': f'Task {task_id} not found'}
            return
        
        if task.status != UITaskStatus.WAITING_INPUT:
            yield {'type': 'error', 'message': 'Task is not waiting for input'}
            return
        
        coordinator = task.coordinator
        if not coordinator:
            yield {'type': 'error', 'message': 'No coordinator for task'}
            return
        
        task.status = UITaskStatus.RUNNING
        task.pending_question = None
        
        # Provide the answer
        coordinator.provide_input('clarification', answer)
        
        # Resume workflow
        progress_queue = self.progress_queues[task_id]
        
        def resume_workflow():
            try:
                for update in coordinator.resume_workflow():
                    progress_queue.put(update.to_dict())
            except Exception as e:
                progress_queue.put({'type': 'error', 'message': str(e)})
            finally:
                progress_queue.put(None)
        
        thread = Thread(target=resume_workflow)
        thread.start()
        
        while True:
            update = progress_queue.get()
            
            if update is None:
                break
            
            if update.get('type') == 'complete':
                task.status = UITaskStatus.COMPLETED
                task.completed_at = datetime.now()
                task.output_path = update.get('data', {}).get('project_dir')
            
            if update.get('type') == 'question':
                task.status = UITaskStatus.WAITING_INPUT
                task.pending_question = update.get('data', {})
            
            yield update
        
        thread.join(timeout=5)
    
    def get_task(self, task_id: str) -> Optional[UITask]:
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    def get_task_dict(self, task_id: str) -> Optional[Dict]:
        """Get task as dictionary"""
        task = self.tasks.get(task_id)
        if not task:
            return None
        
        return {
            'task_id': task.task_id,
            'description': task.description,
            'workflow_type': task.workflow_type,
            'files': task.files,
            'status': task.status.value,
            'created_at': task.created_at.isoformat(),
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None,
            'output_path': task.output_path,
            'error': task.error,
            'pending_question': task.pending_question
        }
    
    def list_tasks(self, status: Optional[UITaskStatus] = None) -> List[Dict]:
        """List all tasks"""
        tasks = list(self.tasks.values())
        
        if status:
            tasks = [t for t in tasks if t.status == status]
        
        return [self.get_task_dict(t.task_id) for t in tasks]
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        if task.status in [UITaskStatus.PENDING, UITaskStatus.RUNNING, UITaskStatus.WAITING_INPUT]:
            task.status = UITaskStatus.CANCELLED
            task.completed_at = datetime.now()
            
            if task.coordinator:
                task.coordinator.cancel()
            
            return True
        
        return False


# Global service instance
_swarm_ui_service: Optional[SwarmUIService] = None


def get_swarm_ui_service(config: Optional[Dict] = None) -> SwarmUIService:
    """Get or create global swarm UI service"""
    global _swarm_ui_service
    
    if _swarm_ui_service is None:
        _swarm_ui_service = SwarmUIService(config)
    
    return _swarm_ui_service
