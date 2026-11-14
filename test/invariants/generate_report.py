"""Generate invariant test report for CI integration.

Usage:
    python -m test.invariants.generate_report
Output:
    test\\invariants\\invariant_report.json
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    """Run invariant tests and generate JSON report."""
    report_path = Path(__file__).parent / "invariant_report.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "test\\invariants\\",
            "-v",
            "--tb=short",
            "-q",
        ],
        capture_output=True,
        text=True,
    )

    passed = result.returncode == 0
    lines = result.stdout.splitlines()
    summary_line = [line for line in lines if "passed" in line or "failed" in line]

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "all_passed": passed,
        "exit_code": result.returncode,
        "summary": summary_line[-1] if summary_line else "Unknown",
        "invariants": {
            "I1_no_side_effects": "PASS" if passed else "UNKNOWN",
            "I2_unit_consistency": "PASS" if passed else "UNKNOWN",
            "I3_conservation": "PASS" if passed else "UNKNOWN",
            "I4_reversibility": "PASS" if passed else "UNKNOWN",
            "I5_commutativity": "PASS" if passed else "UNKNOWN",
            "I6_dt_scaling": "PASS" if passed else "UNKNOWN",
            "I7_bounds_clipping": "PASS" if passed else "UNKNOWN",
        },
        "stdout": result.stdout,
        "stderr": result.stderr,
    }

    report_path.write_text(json.dumps(report, indent=2))
    print(f"Report written to {report_path}")

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
