# dLLM Dependency Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the project dependency strategy internally consistent so `lm_eval` support no longer collides ambiguously with the currently resolved `omegaconf` / `antlr4` combination.

**Architecture:** Treat dependency alignment as a packaging problem, not a runtime hack. First inspect upstream version constraints and prove whether a newer `omegaconf` can coexist with `antlr4 4.11`, then update `pyproject.toml` to encode the chosen strategy, and finally synchronize `README.md` so new environments install dependencies in the intended way.

**Tech Stack:** `pyproject.toml`, `README.md`, `pip`, Python packaging metadata, targeted install/verification commands.

---

## File Map

**Primary packaging file**
- Modify: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/pyproject.toml`

**Documentation file**
- Modify if strategy changes installation instructions: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/README.md`

**Design references**
- Read-only reference: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/docs/superpowers/specs/2026-03-27-dependency-alignment-design.md`

**Explicitly out of scope**
- Do not modify: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow/sampler.py`

### Task 1: Verify Upstream Constraint Feasibility

**Files:**
- Modify: none

- [ ] **Step 1: Record the current conflicting environment state**

Run:

```bash
python -m pip show omegaconf antlr4-python3-runtime lm_eval
```

Expected: current environment shows `omegaconf 2.3.0`, `antlr4-python3-runtime 4.11.0`, and `lm_eval` installed, confirming the conflict is between the resolved environment and package metadata.

- [ ] **Step 2: Query available `omegaconf` versions**

Run:

```bash
python -m pip index versions omegaconf
```

Expected: pip prints the available `omegaconf` releases so you can test whether a newer one exists beyond `2.3.0`.

- [ ] **Step 3: Inspect the newest candidate metadata without editing the repo**

Run:

```bash
python -m pip download --no-deps "omegaconf>=2.4" -d /tmp/omegaconf_probe || python -m pip download --no-deps "omegaconf==2.4.0" -d /tmp/omegaconf_probe
```

Expected: either a newer release downloads, or you learn that no `2.4+` candidate exists.

- [ ] **Step 4: Read the downloaded package metadata to inspect its `antlr4` requirement**

Run:

```bash
python - <<'PY'
from pathlib import Path
import zipfile, tarfile

root = Path('/tmp/omegaconf_probe')
files = sorted(root.iterdir())
assert files, 'no downloaded omegaconf artifact'
artifact = files[-1]
print('ARTIFACT', artifact.name)
if artifact.suffix == '.whl':
    with zipfile.ZipFile(artifact) as zf:
        for name in zf.namelist():
            if name.endswith('METADATA'):
                text = zf.read(name).decode('utf-8', errors='ignore')
                for line in text.splitlines():
                    if 'antlr4' in line.lower() or line.startswith('Version:'):
                        print(line)
                break
else:
    with tarfile.open(artifact) as tf:
        for member in tf.getmembers():
            if member.name.endswith('PKG-INFO'):
                text = tf.extractfile(member).read().decode('utf-8', errors='ignore')
                for line in text.splitlines():
                    if 'antlr4' in line.lower() or line.startswith('Version:'):
                        print(line)
                break
PY
```

Expected: you see the candidate `omegaconf` version and its declared `antlr4-python3-runtime` requirement.

- [ ] **Step 5: Decide the packaging branch**

Decision rule:

```text
If a newer omegaconf release no longer pins antlr4 to 4.9.*, use Task 2A.
If it still pins antlr4 incompatibly, use Task 2B.
```

### Task 2A: Implement the Upgrade-`omegaconf` Strategy

**Files:**
- Modify: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/pyproject.toml`
- Modify if needed: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/README.md`

- [ ] **Step 1: Write the failing packaging check in prose and command form**

Document this expected failure before editing:

```text
Current problem: default dependency resolution permits `omegaconf 2.3.0` and `lm_eval` to be selected together, but `omegaconf 2.3.0` still requires `antlr4-python3-runtime==4.9.*` while the installed `lm_eval` math path uses `antlr4 4.11`.
```

Run:

```bash
python -m pip check || true
```

Expected: current environment may report incompatibility or otherwise confirm that the dependency set is not cleanly resolved.

- [ ] **Step 2: Update `/pyproject.toml` to require the compatible `omegaconf` range**

Edit the dependency line from:

```toml
"omegaconf",
```

to the exact compatible range proven in Task 1, for example:

```toml
"omegaconf>=2.4.0",
```

Use the real verified floor, not a guessed number.

- [ ] **Step 3: Keep `lm_eval` in default dependencies unchanged**

Ensure the final relevant dependency block has this shape:

```toml
"omegaconf>=<verified-version>",
"tqdm",
"matplotlib",
"pytest",
"rich",
"lm_eval",
```

Do not add `math_verify` here unless Task 1 proved the full chain is compatible.

- [ ] **Step 4: Update README installation text only if the install story changed**

If the upgrade strategy means the optional evaluation setup is still accurate, keep `/README.md` unchanged.
If you need to clarify version expectations, adjust the section around line 105 to read like:

```md
### (optional) Evaluation setup

The default project dependencies include `lm_eval` support. For extra evaluation features provided by the vendored harness, initialize and install the submodule explicitly:
```

- [ ] **Step 5: Verify the packaging metadata matches the strategy**

Run:

```bash
python - <<'PY'
import tomllib
from pathlib import Path
data = tomllib.loads(Path('/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/pyproject.toml').read_text())
deps = data['project']['dependencies']
for dep in deps:
    if dep.startswith('omegaconf') or dep.startswith('lm_eval'):
        print(dep)
PY
```

Expected: output shows the explicit upgraded `omegaconf` constraint and the retained `lm_eval` dependency.

### Task 2B: Implement the Split-`math`-extra Strategy

**Files:**
- Modify: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/pyproject.toml`
- Modify: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/README.md`

- [ ] **Step 1: Write the failing packaging premise for the split strategy**

Document the reason for fallback:

```text
The newest compatible `omegaconf` candidate still does not relax the `antlr4` pin enough to coexist cleanly with `lm_eval[math]`, so math-specific evaluation dependencies must move out of the default environment.
```

- [ ] **Step 2: Keep only base `lm_eval` in default dependencies**

Ensure the default dependency block in `/pyproject.toml` remains:

```toml
"omegaconf",
"tqdm",
"matplotlib",
"pytest",
"rich",
"lm_eval",
```

Do not add `math_verify` or `antlr4-python3-runtime==4.11` to default dependencies.

- [ ] **Step 3: Add a dedicated optional evaluation extra**

Add a new extra under `[project.optional-dependencies]` in this shape:

```toml
eval = [
    "lm_eval[ifeval,math]",
]
```

If you prefer a more explicit name such as `eval_math`, use it consistently in both `pyproject.toml` and `README.md`.

- [ ] **Step 4: Update README installation instructions to match the split strategy**

Replace the optional evaluation setup section with wording like:

```md
### (optional) Evaluation setup

The base install includes core `lm_eval` support used by dLLM evaluation code.
If you need the vendored harness plus IFEval / Math extras, run:

```bash
git submodule update --init --recursive
pip install -e ".[eval]"
```
```

If the repository still requires editable installation of the vendored harness specifically, document the exact two-step sequence clearly.

- [ ] **Step 5: Verify that default and optional dependency layers are distinct**

Run:

```bash
python - <<'PY'
import tomllib
from pathlib import Path
data = tomllib.loads(Path('/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/pyproject.toml').read_text())
print('DEFAULT')
for dep in data['project']['dependencies']:
    if dep.startswith('omegaconf') or dep.startswith('lm_eval') or 'math_verify' in dep:
        print(dep)
print('OPTIONAL')
for name, deps in data['project']['optional-dependencies'].items():
    if any('lm_eval' in dep or 'math' in dep.lower() for dep in deps):
        print(name, deps)
PY
```

Expected: default dependencies keep base `lm_eval` only, and the math-enabled path lives under a named optional extra.

### Task 3: Reconcile Local Environment and Verify the Chosen Strategy

**Files:**
- Modify only if needed: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/pyproject.toml`
- Modify only if needed: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/README.md`

- [ ] **Step 1: Reinstall the local project metadata according to the chosen strategy**

If Task 2A was chosen, run:

```bash
python -m pip install -e .
```

If Task 2B was chosen and you want to validate the optional path too, run sequentially:

```bash
python -m pip install -e .
python -m pip install -e ".[eval]"
```

Use the exact extra name you introduced.

- [ ] **Step 2: Check the resolved package set**

Run:

```bash
python -m pip show omegaconf antlr4-python3-runtime lm_eval
```

Expected: the versions printed align with the chosen strategy and no longer represent an unexplained contradiction.

- [ ] **Step 3: Run `pip check` as the packaging health signal**

Run:

```bash
python -m pip check
```

Expected: exit code 0. If it still reports a dependency conflict, return to the strategy decision instead of papering over the warning.

- [ ] **Step 4: Run a targeted import verification**

Run:

```bash
python - <<'PY'
import dllm
from dllm.core.eval.base import BaseEvalHarness
print('IMPORT_OK', BaseEvalHarness.__name__)
PY
```

Expected: prints `IMPORT_OK BaseEvalHarness` without import errors.

- [ ] **Step 5: Commit**

```bash
git add /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/pyproject.toml /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/README.md /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/docs/superpowers/specs/2026-03-27-dependency-alignment-design.md /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/docs/superpowers/plans/2026-03-27-dependency-alignment.md
git commit -m "update dependency strategy for lm_eval"
```

Expected: one commit covering the packaging alignment and its supporting design/plan docs.
