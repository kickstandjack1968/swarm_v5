#!/usr/bin/env python3
"""Non-interactive test runner — drives the swarm on a single request."""
import sys
import os
import io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from swarm_coordinator_v2 import SwarmCoordinator

# Provide auto-answers for clarification prompts
answers = """Use psutil for real-time system metrics collection (CPU, memory, disk, network I/O).
Collect metrics every 10 seconds, store in SQLite.
Use configurable thresholds: WARNING at 80%, CRITICAL at 95% for resource metrics.
Remediation: log warnings, send alerts, and for critical issues attempt service restart via subprocess.
Retain 30 days of data, auto-purge older records. Expect ~100MB/day uncompressed.
Use SQLite WAL mode for concurrent reads. Single writer is fine.
FastAPI endpoints should respond within 200ms for dashboard queries.
Skip malformed metrics with a warning log, continue collecting others.
TLS not required for this version, plain HTTP is fine.
Only use Isolation Forest, EWMA, Z-Score as specified. Use scikit-learn and numpy.
Generate log-based alerts. No email/SMS for this version.
Python 3.10+, no other constraints.
DONE
"""

sys.stdin = io.StringIO(answers)

config_file = "config/config_v2.json"
coordinator = SwarmCoordinator(config_file=config_file)

request = (
    "Create a monitoring and maintenance tool that tracks system health "
    "(CPU, memory, disk, network), detects anomalies using machine learning "
    "(Isolation Forest, EWMA, Z-Score), and proactively addresses issues with "
    "automated remediation. Use FastAPI for the web dashboard, SQLite for "
    "metrics storage with AES-256 encryption, and psutil for system metrics. "
    "Include Prometheus-compatible /metrics endpoint."
)

print(f"Request: {request[:80]}...")
print("=" * 80)
coordinator.run_workflow(request, workflow_type="standard")
print("\n" + "=" * 80)
print("DONE")
