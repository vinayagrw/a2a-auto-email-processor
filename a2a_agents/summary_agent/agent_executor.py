"""
Summary Agent Executor.

This module contains the executor for the Summary Agent, which handles the
actual execution of summary generation tasks.
"""

import logging
from typing import Any, Dict, Optional, override, AsyncGenerator



from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent
)
from a2a.utils import new_agent_text_message, new_task, new_text_artifact

from a2a_agents.summary_agent.agent import SummaryAgent

logger = logging.getLogger(__name__)


logger = logging.getLogger(__name__)

class SummaryAgentExecutor(AgentExecutor):
    """Executor for the Summary Agent."""
        
    def __init__(self, **kwargs):
        """Initialize the SummaryAgentExecutor."""
        super().__init__(**kwargs)
        self.agent = SummaryAgent()
    
    @override
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue
    ) -> None:
        """
        Execute the summary generation task.
        
        Args:
            context: The execution context containing the task data
            event_queue: Queue for sending progress updates
        """
        query = context.get_user_input()
        task = context.current_task
        logger.info(f"\nExecuting summary generation task: {query}")

        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)
        
        try:


            async for event in self.agent.generate_summary(query, task.context_id):
                task_state = TaskState(event.get('task_state', TaskState.working))
                
                if event.get('is_task_complete', False):
                    # Handle task completion
                    await event_queue.enqueue_event(
                        TaskArtifactUpdateEvent(
                            append=False,
                            context_id=task.context_id,
                            task_id=task.id,
                            last_chunk=True,
                            artifact=new_text_artifact(
                                name='email_summary_result',
                                description='Result of email summary',
                                text=event.get('content', ''),
                            ),
                        )
                    )
                    await event_queue.enqueue_event(
                        TaskStatusUpdateEvent(
                            status=TaskStatus(state=task_state),
                            final=True,
                            context_id=task.context_id,
                            task_id=task.id,
                        )
                    )
                else:
                    # Handle progress updates
                    await event_queue.enqueue_event(
                        TaskStatusUpdateEvent(
                            status=TaskStatus(
                                state=task_state,
                                message=new_agent_text_message(
                                    event.get('content', ''),
                                    task.context_id,
                                    task.id,
                                ),
                            ),
                            final=task_state in {
                                TaskState.input_required,
                                TaskState.failed,
                                TaskState.unknown,
                            },
                            context_id=task.context_id,
                            task_id=task.id,
                        )
                    )
                    
        except Exception as e:
            logger.error(f"Error in summary generation: {e}", exc_info=True)
            await self._handle_execution_error(
                TaskUpdater(event_queue, task.id, task.context_id) if task else None,
                task,
                str(e)
            )

    @override
    async def cancel(self, request: RequestContext, event_queue: EventQueue) -> Task | None:
        """Handle task cancellation.
        
        Args:
            request: The cancellation request context
            event_queue: The event queue for status updates
            
        Returns:
            The cancelled task or None if not found
        """
        task = request.current_task
        if not task:
            logger.warning('No active task to cancel')
            return None
            
        logger.warning(f'Cancellation requested for task {task.id}')
        # TODO: Implement proper cancellation logic
        return None
        
    async def _handle_progress_update(
        self, updater: TaskUpdater, task: Task, update: Dict[str, Any]
    ) -> None:
        """Handle progress update from the agent."""
        message = update.get('updates', 'Processing email...')
        progress = update.get('progress_percent', 0)
        
        logger.debug(f'Task {task.id} progress: {message} ({progress}%)')
        
        await updater.update_status(
            TaskState.working,
            message=new_agent_text_message(message, task.context_id, task.id)
        )
    
    async def _handle_task_completion(
        self, updater: TaskUpdater, task: Task, update: Dict[str, Any]
    ) -> None:
        """Handle successful task completion."""
        if update.get('is_error', False):
            await self._handle_task_error(updater, task, update)    
            return
            
        result = update.get('result', {})
        logger.info(f'Task {task.id} completed successfully')
        
        final_message = new_agent_text_message(
            update.get('final_message_text', 'Email processing completed'),
            task.context_id,
            task.id
        )
        
        if 'artifacts' in result:
            await updater.add_artifact(result['artifacts'])
            
        await updater.complete(final_message)
    
    async def _handle_task_error(
        self, updater: TaskUpdater, task: Task, update: Dict[str, Any]
    ) -> None:
        """Handle task error."""
        error_msg = update.get('error', 'Unknown error occurred')
        logger.error(f'Task {task.id} failed: {error_msg}')
        
        await updater.fail(new_agent_text_message(
            f'Error processing email: {error_msg}',
            task.context_id,
            task.id
        ))
    
    async def _handle_execution_error(
        self, updater: TaskUpdater, task: Task, error: Exception
    ) -> None:
        """Handle execution errors."""
        logger.error(f'Error in response generation task {task.id}: {error}', exc_info=True)
        
        if updater:
            await updater.failed(new_agent_text_message(
                f'Failed to generate response: {error}',
                task.context_id if task else None,
                task.id if task else None
            ))