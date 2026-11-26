"""AAROHAN AI - Structured Tool Execution & Verification Model.

In accordance with AAROHAN AI - Execution & Verification Engine 1.0:
Every tool returns a structured execution result containing:
- task_id: Unique identifier for the execution turn
- tool_used: Exact tool invoked
- execution_status: GENERATED, EXECUTED, VALIDATED, FAILED, or NOT_SUPPORTED
- output_location: Absolute or relative file path / URL
- validation_status: VALIDATED, UNVALIDATED, or FAILED
- validation_details: Detailed verification evidence (e.g. reopened row count, column match)
- error_details: Error diagnostic if failed
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class ExecutionStatus(str, enum.Enum):
    GENERATED = "GENERATED"
    EXECUTED = "EXECUTED"
    VALIDATED = "VALIDATED"
    FAILED = "FAILED"
    NOT_SUPPORTED = "NOT_SUPPORTED"


@dataclass
class ExecutionResult:
    task_id: str
    tool_used: str
    execution_status: ExecutionStatus
    validation_status: str  # "VALIDATED", "UNVALIDATED", "FAILED"
    output_location: Optional[str] = None
    output_preview: Optional[Any] = None
    validation_details: Optional[Dict[str, Any]] = None
    error_details: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        tool_used: str,
        status: ExecutionStatus = ExecutionStatus.EXECUTED,
        task_id: Optional[str] = None,
        validation_status: str = "VALIDATED",
        output_location: Optional[str] = None,
        output_preview: Optional[Any] = None,
        validation_details: Optional[Dict[str, Any]] = None,
        error_details: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        return cls(
            task_id=task_id or f"task_{uuid.uuid4().hex[:10]}",
            tool_used=tool_used,
            execution_status=status,
            validation_status=validation_status,
            output_location=output_location,
            output_preview=output_preview,
            validation_details=validation_details or {},
            error_details=error_details,
            metadata=metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "tool_used": self.tool_used,
            "execution_status": self.execution_status.value,
            "validation_status": self.validation_status,
            "output_location": self.output_location,
            "output_preview": self.output_preview,
            "validation_details": self.validation_details,
            "error_details": self.error_details,
            "metadata": self.metadata,
        }
