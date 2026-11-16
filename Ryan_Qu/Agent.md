# Code Quality and Readability

These instructions apply to all source code in this ML research repository.

The goal is human-readable, straightforward research code.

Prioritize:
1. Functional correctness.
2. Human readability.
3. Simplicity.
4. Maintainability.
5. Minimal unnecessary engineering.

Do not optimize for the smallest number of lines.

# Scope instruction: without the user explicitly demanding repo cleanup, every instrution shuold refer to 1-3 in the following, and ignore 4 - the end unless the user mentions cleanup.

## 1. Human-Readable Code

### Naming

- Use meaningful names that reveal the purpose of variables and functions.
- Prefer descriptive names over unexplained abbreviations.
- Keep mathematical and ML conventions where appropriate:
  `x`, `y`, `W`, `H`, `lr`, `epoch`, `batch_size`, etc.
- Avoid meaningless names such as `tmp`, `obj`, or `thing` when more descriptive names are available.
- Maintain consistent naming throughout related code.

### Control Flow

- Prefer straightforward, explicit control flow.
- Avoid deeply nested conditionals when simple guard clauses improve readability.
- Avoid complicated one-line expressions.
- Avoid nested ternary expressions.
- Avoid excessively long chains of method calls.
- Use intermediate variables when they make computations easier to follow.

Readable code is more important than clever code.

Do not compress multiple meaningful operations into one line merely to reduce line count.

### Functions

- Give functions a clear, identifiable responsibility.
- Keep closely related logic together.
- Split functions only when doing so improves comprehension.
- Do not create unnecessary helper functions for trivial expressions.
- Avoid excessive abstraction and wrapper functions.
- Prefer direct implementations over elaborate class hierarchies.

Do not introduce design patterns without a concrete need.

### Comments and Documentation

- Explain why an operation exists when the reason is not obvious.
- Explain important mathematical assumptions or algorithmic decisions.
- Use short comments for nontrivial transformations.
- Do not comment every line.
- Do not restate obvious Python syntax.
- Avoid excessive docstrings that repeat function names and parameters.
- Preserve useful research notes and scientific reasoning.

### Formatting

For Python code:

- Follow established PEP 8 conventions.
- Use consistent four-space indentation.
- Use consistent whitespace and blank lines.
- Keep imports organized and readable.
- Format long function calls and data structures across multiple lines.
- Avoid dense, difficult-to-read expressions.
- Preserve existing repository conventions where reasonable.

Ruff is the preferred optional formatter:

    ruff format path/to/file.py

For verification without modifying files:

    ruff format --check path/to/file.py

Only format files relevant to the assigned task.

Do not automatically format the entire repository.
Do not add new mandatory linting infrastructure.
Do not run automatic lint fixes that could change behavior.

## 2. Avoid Excessive Defensive Engineering

Research code should not be burdened with unnecessary validation, auditing, or approval systems.

Every validation should prevent a concrete, realistic failure.

Before keeping a check, consider:

- What failure does it prevent?
- Is that failure relevant to this operation?
- Has the same property already been verified upstream?
- Does this check belong in the current worker?
- Would removing it compromise actual functionality?

### Remove Redundant Hashing

Avoid:

- Repeated SHA-256 calculations on the same files.
- Repeated file fingerprint comparisons.
- Hashing outputs solely to confirm that they were generated.
- Repeated hash-manifest creation across pipeline stages.
- Duplicate integrity checks without a real integrity requirement.

Checksums remain appropriate for genuine transfer integrity,
security, reproducibility, or final artifact delivery.

### Remove Unnecessary Pixel Comparisons

Avoid:

- Regenerating images solely for pixel-by-pixel verification.
- Repeatedly comparing unchanged images.
- Reconstructing preprocessing outputs just to compare them.
- Decoding the same dataset repeatedly for redundant verification.

Preserve pixel comparisons that are part of actual algorithms,
evaluation metrics, or required scientific tests.

### Avoid Cross-Stage Auditing

A worker should perform its own responsibility.

Do not make downstream workers repeatedly reconstruct and
verify everything performed by upstream workers.

Prefer:

    Input
      -> Essential Input Checks
      -> Core Transformation
      -> Output

Avoid:

    Input
      -> Validation
      -> Reconstruction
      -> Repeated Validation
      -> Hash Audit
      -> Additional Verification
      -> Output

Trust documented upstream contracts unless there is a
concrete reason to validate data at a new boundary.

### Avoid Redundant Dataset Validation

Remove unnecessary:

- Repeated recursive dataset scans.
- Duplicate image inventory checks.
- Multiple label-validation passes for the same invariant.
- Repeated metadata consistency audits.
- Redundant geometry reconstruction.
- Unused verification reports.

Preserve checks necessary to prevent incorrect training,
invalid annotations, data leakage, and destructive operations.

### Avoid Unnecessary Infrastructure

Do not introduce unrequested:

- Approval systems.
- Reviewer tracking.
- Audit frameworks.
- Validation frameworks.
- Complex wrapper classes.
- Speculative compatibility layers.
- Redundant metadata registries.

Do not replace removed complexity with new complexity.

## 3. Preserve Repository Organization

Code-quality work must not modify repository organization.

Do not:

- Move or rename directories.
- Move or rename source files.
- Reorganize packages.
- Delete historical experiment folders.
- Delete old notebooks.
- Modify datasets or annotations.
- Delete checkpoints or research results.
- Rewrite existing project architecture.

Perform code changes in place.

Respect unrelated and uncommitted changes.

## Note: without the user explicitly requiring repo cleanup, ignore the following instruction.


## 4. Preserve Scientific Functionality 

Do not change the following during code-quality cleanup:

- ML algorithms.
- Mathematical computations.
- Model architectures.
- Training behavior.
- Evaluation metrics.
- Data transformations.
- Dataset contents.
- Ground-truth annotations.
- Public interfaces.
- Experiment configurations.
- Random seed behavior where reproducibility matters.

Do not introduce new features during readability cleanup.

For defense-cleaning specifically, removing a redundant audit
may intentionally remove its associated failure condition.
This is permitted only when the audit is demonstrably unnecessary
and normal pipeline functionality remains unchanged.

Do not remove genuine security or correctness checks merely
to make code shorter.

## 5. Mandatory Smoke Tests

Every functional refactoring requires verification.

### Before Editing

1. Understand the affected function or worker.
2. Identify its main input-transformation-output behavior.
3. Run a lightweight representative smoke test.
4. Record the baseline behavior.

Use small fixtures or temporary test data.

Do not run expensive GPU training unless explicitly requested.

### After Editing

1. Rerun the same smoke test.
2. Confirm successful execution.
3. Compare relevant outputs against the baseline.
4. Confirm that core functionality remains unchanged.

For deterministic operations, compare representative output values.

For nondeterministic ML operations, compare meaningful
functional invariants rather than requiring bit-identical GPU outputs.

Do not modify tests merely to conceal a regression.

If a test fails, fix or revert the responsible change and rerun it.

If testing is impossible, report the affected functionality as
UNVERIFIED rather than claiming success.

## 6. Strict Stopping Rule

Once the requested cleanup is complete and relevant smoke tests pass:

STOP.

Do not continue into:

- Unrelated optimization.
- Additional code-style changes.
- Architecture redesign.
- Directory restructuring.
- New testing frameworks.
- New validation systems.
- Further speculative refactoring.

Do not repeatedly search for new improvements after the assigned
scope has been successfully completed.

The objective is not perfect code.

The objective is readable, functioning code without unnecessary
engineering overhead.

## 7. Completion Report

After code-quality work, report briefly:

- Source files modified.
- Main readability improvements.
- Redundant defensive mechanisms removed, if applicable.
- Important functionality preserved.
- Smoke-test commands and actual results.
- Final status: PASS, FAIL, or UNVERIFIED.

Do not automatically commit or push changes.
