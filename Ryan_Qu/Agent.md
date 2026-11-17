# Code Quality and Readability

These instructions apply to all source code and newly created research artifacts in this ML research repository.

The goal is human-readable, straightforward research code and an organized repository that a researcher can understand without reconstructing previous conversations.

Prioritize:

1. Functional correctness.
2. Human readability.
3. Simplicity.
4. Maintainability.
5. Minimal unnecessary engineering.

Do not optimize for the smallest number of lines or the shortest possible file and directory names.

## Scope and Execution Boundaries — READ FIRST

**Sections 1–3 are always on**, but apply only within the scope of the user's actual request. They govern how new code, files, directories, and documentation should be created and how existing work should be protected.

**Sections 4–7 are conditional**. Follow them only when the user explicitly requests code cleanup, simplification, or behavior-preserving refactoring. Their presence does not authorize independent cleanup.

- Do not initiate cleanup, restructuring, or refactoring automatically.
- Do not interpret guidance to *avoid adding* defensive engineering as permission to *remove existing* defensive engineering.
- Do not make unrelated changes, including to historical experiments, datasets, or documentation.
- Formatting requests alone do not authorize code cleanup or defensive-check removal.
- Apply the new naming and organization rules to **future work only**. Do not retroactively rename or reorganize existing files or directories unless explicitly requested.

---

# Part I — Always-On Policies

## 1. Human-Readable Code

### Naming

- Use meaningful names that reveal the purpose of variables and functions.
- Prefer descriptive names over unexplained abbreviations.
- Keep mathematical and ML conventions where appropriate: `x`, `y`, `W`, `H`, `lr`, `epoch`, `batch_size`, etc.
- Avoid meaningless names such as `tmp`, `obj`, or `thing` when more descriptive names are available.
- Maintain consistent naming throughout related code.
- Prioritize clarity over brevity; do not shorten names at the expense of understanding.

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
- Format long function calls and data structures across multiple lines when needed for readability.
- Avoid dense, difficult-to-read expressions.
- Preserve existing repository conventions where reasonable.

Ruff is the preferred optional formatter:

```bash
ruff format path/to/file.py
```

For verification without modifying files:

```bash
ruff format --check path/to/file.py
```

Only format files relevant to the assigned task.

Do not automatically format the entire repository. Do not add new mandatory linting infrastructure. Do not run automatic lint fixes that could change behavior.

## 2. Avoid Excessive Defensive Engineering

Research code should not be burdened with unnecessary validation, auditing, or approval systems.

**For ordinary development, this section governs checks and infrastructure being introduced or changed as part of the assigned task. It does not authorize removal of existing checks.**

Every newly introduced validation should prevent a concrete, realistic failure.

Before adding a check, consider:

- What failure does it prevent?
- Is that failure relevant to this operation?
- Has the same property already been verified upstream?
- Does this check belong in the current worker?
- Is its complexity justified by the actual failure risk?

### Avoid Redundant Hashing

Do not introduce unnecessary:

- Repeated SHA-256 calculations on the same files.
- Repeated file fingerprint comparisons.
- Hashing outputs solely to confirm that they were generated.
- Repeated hash-manifest creation across pipeline stages.
- Duplicate integrity checks without a real integrity requirement.

Checksums remain appropriate for genuine transfer integrity, security, reproducibility, or final artifact delivery.

### Avoid Unnecessary Pixel Comparisons

Do not introduce unnecessary:

- Regenerating images solely for pixel-by-pixel verification.
- Repeatedly comparing unchanged images.
- Reconstructing preprocessing outputs just to compare them.
- Decoding the same dataset repeatedly for redundant verification.

Preserve pixel comparisons that are part of actual algorithms, evaluation metrics, or required scientific tests.

### Avoid Cross-Stage Auditing

A worker should perform its own responsibility.

Do not make downstream workers repeatedly reconstruct and verify everything performed by upstream workers.

Prefer:

```text
Input
  -> Essential Input Checks
  -> Core Transformation
  -> Output
```

Avoid:

```text
Input
  -> Validation
  -> Reconstruction
  -> Repeated Validation
  -> Hash Audit
  -> Additional Verification
  -> Output
```

Trust documented upstream contracts unless there is a concrete reason to validate data at a new boundary.

### Avoid Redundant Dataset Validation

Do not introduce unnecessary:

- Repeated recursive dataset scans.
- Duplicate image inventory checks.
- Multiple label-validation passes for the same invariant.
- Repeated metadata consistency audits.
- Redundant geometry reconstruction.
- Unused verification reports.

Preserve checks necessary to prevent incorrect training, invalid annotations, data leakage, and destructive operations.

### Avoid Unnecessary Infrastructure

Do not introduce unrequested:

- Approval systems.
- Reviewer tracking.
- Audit frameworks.
- Validation frameworks.
- Complex wrapper classes.
- Speculative compatibility layers.
- Redundant metadata registries.

Do not replace simple work with new complexity.

**Removing existing redundant checks is a cleanup task and requires explicit authorization under Section 4.**

## 3. Repository and Experiment Organization

### 3.1 Preserve Existing Work

Do not, without an explicit request:

- Move or rename existing directories.
- Move or rename existing source files.
- Reorganize packages.
- Delete historical experiment folders.
- Delete old notebooks.
- Modify existing datasets or annotations.
- Delete checkpoints or research results.
- Rewrite existing project architecture.

Perform relevant code changes in place. Respect unrelated and uncommitted changes.

Creating new files or directories needed for an explicitly requested task is allowed; it is not permission to reorganize unrelated existing work.

### Dataset inputs and experiment outputs

All input datasets used by the project, including original datasets and prepared
model inputs, belong under `Ryan_Qu/shared_dataset/`. For each dataset, organize
its files under a descriptive dataset directory with:

- `raw/`: immutable originals, preserved in their original state.
- `canonical/`: verified, orientation-corrected or otherwise prepared inputs.
- `manifest/`: dataset definitions, source mappings, labels, splits, and other
  metadata needed to explain which inputs are included.

Do not label a dataset as verified unless the relevant verification was actually
performed; record its verification status and applicable transformations.
Training records, model outputs, checkpoints, and other experiment-produced
artifacts remain in their respective experiment/output directories. This policy
describes the intended organization; inventory and plan migrations first, and
do not move, delete, or restructure existing datasets until the user explicitly
directs that particular cleanup.

### 3.2 Human-Readable File and Directory Names

- **Prioritize semantic clarity over brevity.** Never sacrifice human understanding simply to make a name shorter.
- Use descriptive, natural English names that explain the purpose of a file, experiment, or output.
- Names with **up to 6–7 meaningful words** are acceptable when they genuinely improve understanding. Avoid excessive verbosity, but do not remove essential context.
- Aim for names that a researcher can understand at a glance, even weeks or months later.
- Avoid ambiguous names such as `test2`, `unseen_100`, `final_v3`, or `run_new` when they do not communicate the actual purpose.
- Prefer an experiment's purpose or methodology over isolated sample counts, parameter values, or implementation details. Keep such technical details in configuration or metadata when appropriate.
- Follow existing project naming conventions when they are meaningful. Retain required identifiers or timestamps if the repository already uses them; combine them with descriptive names where possible.
- Use names faithful to the actual task. Do not label an evaluation as a manual inspection unless manual inspection was truly performed.

### 3.3 Organize New Experiments for Human Understanding

- Inspect the relevant existing directory structure before choosing where new experiment files belong.
- Reuse existing experiment categories and directory conventions instead of inventing competing structures.
- Keep reusable source code distinct from generated experiment outputs where the project structure permits.
- Avoid unnecessary directory nesting, fragmentation, or duplicate result folders.
- Make the purpose of each experiment and its relationship to related experiments easy to find.
- Do not introduce new tracking frameworks or elaborate metadata systems without a concrete need.

### 3.4 Experiment Documentation

For each substantial new experiment, briefly document:

- **Purpose:** Why was it conducted?
- **Method:** What was tested or changed?
- **Results:** What actually happened, with links or paths to relevant outputs?
- **Conclusion:** What can be learned from the results? If not yet evaluated, say so rather than inventing findings.

Use the repository's existing documentation pattern, such as an experiment README or a shared experiment log. Do not create duplicate documentation simply to satisfy a template.

Keep documentation concise, natural, and understandable. Avoid cryptic abbreviations, needless jargon, and descriptions that require reading source code to decipher.

### 3.5 Before Finishing

For newly created or modified files, directories, and documentation within the current task, ask:

> Could another researcher understand what was done simply by browsing these artifacts, without inspecting the implementation or reconstructing the conversation?

If not, improve the **new work within scope** before finishing. This check does not authorize cleanup, renaming, or reorganization of unrelated or historical artifacts.

---

# Part II — Conditional Cleanup Protocol

**Sections 4–6 apply only when the user explicitly requests cleanup, code simplification, or behavior-preserving refactoring.** Ordinary feature work, experimentation, bug fixes, or formatting alone do not activate them.

## 4. Preserve Scientific Functionality During Cleanup

Do not change the following during code-quality cleanup unless the user explicitly requests it:

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

For defensive-engineering cleanup specifically, removing a redundant audit may intentionally remove its associated failure condition. This is permitted only when the audit is demonstrably unnecessary and normal pipeline functionality remains unchanged.

Do not remove genuine security or correctness checks merely to make code shorter.

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

For nondeterministic ML operations, compare meaningful functional invariants rather than requiring bit-identical GPU outputs.

Do not modify tests merely to conceal a regression.

If a test fails, fix or revert the responsible change and rerun it.

If testing is impossible, report the affected functionality as `UNVERIFIED` rather than claiming success.

## 6. Strict Stopping Rule

Once the requested cleanup is complete and relevant smoke tests pass:

**STOP.**

Do not continue into:

- Unrelated optimization.
- Additional code-style changes.
- Architecture redesign.
- Directory restructuring.
- New testing frameworks.
- New validation systems.
- Further speculative refactoring.

Do not repeatedly search for new improvements after the assigned scope has been successfully completed.

The objective is not perfect code.

The objective is readable, functioning code without unnecessary engineering overhead.

---

# Part III — Task-Dependent Reporting

## 7. Completion Report

After ordinary development, report briefly:

- What was changed and which files were affected.
- Relevant tests actually run and their results.
- Any blockers or limitations.

After explicitly requested code-quality cleanup, also report:

- Main readability improvements.
- Redundant defensive mechanisms removed, if applicable.
- Important functionality preserved.
- Smoke-test commands and actual results.
- Final status: `PASS`, `FAIL`, or `UNVERIFIED`.

Do not automatically commit or push changes.

---

**Final reminder:** The current user request determines what work is authorized. These instructions determine how that work must be performed; they do not independently authorize cleanup or reorganization.
