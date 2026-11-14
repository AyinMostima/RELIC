# Dependency Governance SSOT

This document defines the single-source policy for RELIC dependency security
evidence. A passing gate means that the declared project runtime closure in the
verified RELIC environment was completely enumerated and audited, and that all
reported advisories are either absent or covered by current explicit policy. It
does not mean that every package installed in the developer environment is part
of RELIC or vulnerability-free.

## Canonical inputs and owner

- `pyproject.toml`: the sole runtime dependency declaration.
- `test\audit\dependency_audit_manifest.json`: audit scope and critical package list.
- `test\audit\dependency_skip_allowlist.json`: reviewed package and advisory exceptions.
- `test\audit\dependency_audit_summary.py`: the sole collector and gate owner.
- `analysis_results\dependency_audit_summary.json`: generated evidence; never edit by hand.

RELIC does not require or regenerate `requirements.lock.txt` or `uv.lock` for
this gate. The build pipeline must not impose a rigid dependency pin set.
Wheel, sdist, and evidence-container identity are bound by final artifact
SHA-256 values in the existing G8 package and evidence owners.

## Audit scope

The fixed mode is `project_environment`. The owner:

1. reads `[project].dependencies` from `pyproject.toml`;
2. evaluates PEP 508 markers and requested extras for the current interpreter;
3. follows installed `Requires-Dist` edges to enumerate the complete runtime closure;
4. verifies every declared constraint against the installed distribution version;
5. requires every manifest-declared critical package to be both directly declared and resolved;
6. maps a PEP 440 local version to its public version only for advisory lookup, while retaining the exact installed version in evidence;
7. writes the resolved versions to an automatically deleted temporary input;
8. runs `pip-audit --strict --no-deps --disable-pip` on that temporary snapshot;
9. requires the pip-audit package/version set and digest to exactly match the enumerated closure.

The temporary exact-version snapshot is runtime evidence only. It is not a
source artifact, install input, persistent lock, or build constraint.

## Critical package rule

`torch` is currently critical. It must be directly declared in
`pyproject.toml`, installed in the formal RELIC environment, resolved into the
closure, and returned by pip-audit with the expected public advisory version.

If pip-audit skips any dependency, the release gate fails. A critical-package
skip also increments `critical_skip_without_policy` unless a current
package policy exists. Package policy does not substitute for a completed
advisory scan.

If an advisory cannot be remediated immediately, its exception must be keyed
by the reported advisory ID or alias and contain:

- reason;
- mitigation;
- owner;
- review date.

The review date is valid through that date and expires the next day. A generic
package-level entry never accepts every advisory automatically.

## Fail-closed gate

The generated schema is `2.0.0`. The gate passes only when all conditions hold:

```text
pass =
  project metadata and audit manifest are valid
  and every applicable direct and transitive dependency is installed
  and every installed version satisfies its declaration
  and all critical packages are declared and resolved
  and pip-audit returns valid non-empty JSON with return code 0 or 1
  and return code agrees with the vulnerability payload
  and pip-audit package names, versions, count, and digest exactly match the closure
  and there are no skipped packages
  and there are no policy schema or review-date issues
  and there are no unaccepted vulnerabilities
```

Timeout, malformed or missing JSON, zero dependencies, missing distributions,
constraint violations, marker/extra closure gaps, duplicate or unbound audit
rows, version mismatch, digest mismatch, and return-code contradictions all
fail closed.

Accepted advisory exceptions remain visible security debt. The raw count is
conserved as `accepted_vulnerability_count + vulnerability_count`; consumers
must report both values.

## Current measured environment

The code-state audit generated on 2026-09-27 at 15:03:02 UTC used
`python` and reports the installed runtime closure
in `analysis_results\dependency_audit_summary.json`:

- `pip check` reported no broken requirements;
- 25 direct runtime dependencies expanded to 101 installed runtime dependencies;
- input and audited dependency digests matched exactly;
- critical missing, input integrity, and skipped-package counts were zero;
- pip-audit returned code 0 with zero known vulnerabilities.

The local CUDA build `torch==2.13.0+cu130` is recorded exactly and queried as
public version `2.13.0` for advisory matching. The current executable reports
Python 3.11.13; this gate does not prove Python 3.14 runtime compatibility.

The same RELIC environment provides the development checker `mypy 1.19.1`
with `librt 0.15.0`. The development declarations in `environment.yml` and
`pyproject.toml` require `mypy>=1.19.1`: the previous checker crashed while
analysing `onnx_ir 1.0.0` (mypy issue #20015). Repair the RELIC toolchain
itself instead of bypassing this check in repository configuration. The
installed CUDA build is retained in `requirements.lock.txt`; that inventory
is not a rigid build-time dependency constraint. A fresh interpreter and
`pip check` probe on 2026-09-27 confirmed these versions and no broken
requirements.

## Entry points

Run in the RELIC environment:

```powershell
python -m pip check
python -m test.audit.dependency_audit_summary
python -m test.audit.unified_scientific_gate
```

Focused regression tests:

```powershell
python -m pytest test\audit\dependency_audit_summary.test.py -q
python -m pytest test\audit\unified_scientific_gate.spec.py -q
```

Inspect these generated fields before release:

- `input_report.project_metadata_sha256`;
- direct and resolved dependency counts;
- exact installed/public-version package records;
- input and audited dependency SHA-256 values;
- `critical_input_missing` and `input_integrity_violations`;
- accepted and unaccepted vulnerability counts;
- skipped packages, command, return code, stderr, and error;
- top-level `pass`.

Any dependency declaration, formal environment, policy, interpreter, or
security database change requires regenerating the report. Any source change
also invalidates the prior source fingerprint and G8 evidence bundle.
