# Scope and Deliverables

## Why This Layer Exists

RELIC already has a detailed architecture document and deeper theory and case
materials. What new users usually need first, however, is a shorter entry
path: what the notebook suite is for, what it covers, and what to run first.

This directory fills that gap.

## Scope

The primary tutorial layer is responsible for:

- introducing the `tutorial\primary` notebook suite
- mapping notebooks to real RELIC capabilities
- explaining the intended learning order
- pointing users back to the architecture and theory SSOT documents

It is **not** responsible for duplicating:

- `docs\source\architecture_overview.md`
- the full theory derivations in `tutorial\docs\theory`
- the complete case workflow in `tutorial\case`
- source-level implementation details for every scientific module

## Intended Audience

- first-time RELIC users
- project reviewers
- developers who need a guided refresher before reading core source files

## Deliverables

This release-facing tutorial layer contains:

1. `tutorial\docs\primary`
   - 7 short guide documents
   - one index and six focused reference notes

2. `tutorial\primary`
   - 30 executable notebooks
   - notebook outputs stored directly in each `.ipynb`

3. `reports`
   - execution summaries and audit artifacts generated during verification

## Acceptance Expectations

For distribution, the tutorial package should satisfy the following:

- notebooks execute successfully in the `RELIC` environment
- outputs are saved in the notebooks themselves
- notebook ordering is clear to a new user
- the guide documents are short enough to scan quickly
- architecture and theory references remain explicit
