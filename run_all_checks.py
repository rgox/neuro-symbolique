"""
Comprehensive Platform Check (No external dependencies needed for code analysis)
"""
import re
import ast
from pathlib import Path
import subprocess

ROOT = Path("/home/ben/Bureau/neuro-symbolique")

print("=" * 80)
print("COMPREHENSIVE CODE VERIFICATION - NEURO-SYMBOLIQUE PLATFORM")
print("=" * 80)

# =============================================================================
# 1. FILE STRUCTURE VERIFICATION
# =============================================================================
print("\n1. FILE STRUCTURE VERIFICATION")
print("-" * 80)

required_files = [
    "nesy/__init__.py",
    "nesy/core/memory.py",
    "nesy/core/config.py",
    "nesy/core/telemetry.py",
    "nesy/hal/device.py",
    "nesy/hal/npu/simulator.py",
    "nesy/hal/spu/simulator.py",
    "nesy/hal/cpu/orchestrator.py",
    "nesy/world_model/scene_graph/graph.py",
    "nesy/world_model/scene_graph/layers.py",
    "nesy/world_model/scene_graph/spatial_index.py",
    "nesy/perception/object_detection.py",
    "nesy/perception/feature_extraction.py",
    "nesy/perception/pipeline.py",
]

missing = []
for f in required_files:
    if not (ROOT / f).exists():
        missing.append(f)
        print(f"  ✗ MISSING: {f}")

if not missing:
    print(f"  ✓ All {len(required_files)} required files present")
else:
    print(f"  ✗ {len(missing)} files missing!")

# =============================================================================
# 2. SYNTAX CHECK (All files must be valid Python)
# =============================================================================
print("\n2. SYNTAX VALIDATION")
print("-" * 80)

py_files = list(ROOT.glob("nesy/**/*.py"))
py_files = [f for f in py_files if '__pycache__' not in str(f)]

syntax_errors = []
for py_file in py_files:
    try:
        compile(py_file.read_text(), str(py_file), 'exec')
    except SyntaxError as e:
        syntax_errors.append((py_file, e))

if syntax_errors:
    print(f"  ✗ {len(syntax_errors)} files with syntax errors:")
    for f, e in syntax_errors[:5]:
        print(f"    - {f.relative_to(ROOT)}: {e}")
else:
    print(f"  ✓ All {len(py_files)} files have valid Python syntax")

# =============================================================================
# 3. CODE QUALITY CHECKS
# =============================================================================
print("\n3. CODE QUALITY ANALYSIS")
print("-" * 80)

issues = {
    'bare_except': [],
    'missing_docstring': [],
    'long_lines': [],
    'complex_functions': [],
}

for py_file in py_files:
    content = py_file.read_text()
    lines = content.split('\n')
    
    # Check bare except
    for i, line in enumerate(lines):
        if re.match(r'\s*except\s*:', line):
            issues['bare_except'].append((py_file.name, i+1))
    
    # Check docstrings
    if content.startswith('"""') or content.startswith("'''"):
        pass  # Has module docstring
    elif 'class ' in content or 'def ' in content:
        if '"""' not in content[:100] and "'''" not in content[:100]:
            issues['missing_docstring'].append(py_file.name)
    
    # Check line length
    for i, line in enumerate(lines):
        if len(line) > 120:
            issues['long_lines'].append((py_file.name, i+1, len(line)))

# Print summary
print(f"  Bare except clauses: {len(issues['bare_except'])}")
if issues['bare_except'][:3]:
    for file, line in issues['bare_except'][:3]:
        print(f"    - {file}:{line}")

print(f"  Long lines (>120 chars): {len(issues['long_lines'])}")
if issues['long_lines'][:3]:
    for file, line, length in issues['long_lines'][:3]:
        print(f"    - {file}:{line} ({length} chars)")

if sum(len(v) for v in issues.values()) == 0:
    print("  ✓ No major code quality issues detected")

# =============================================================================
# 4. MODULE DEPENDENCIES
# =============================================================================
print("\n4. MODULE DEPENDENCIES ANALYSIS")
print("-" * 80)

imports = set()
for py_file in py_files:
    content = py_file.read_text()
    # Find imports
    import_lines = [l for l in content.split('\n') if l.strip().startswith(('import ', 'from '))]
    for line in import_lines:
        match = re.match(r'(?:from|import)\s+(\S+)', line)
        if match:
            module = match.group(1).split('.')[0]
            imports.add(module)

external_deps = imports - {'nesy', 'typing', 'dataclasses', 'enum', 'abc', 'collections', 'pathlib', 'os', 'sys', 'time', 're'}
print(f"  External dependencies: {len(external_deps)}")
for dep in sorted(external_deps):
    print(f"    - {dep}")

# =============================================================================
# 5. CODE STATISTICS
# =============================================================================
print("\n5. CODE STATISTICS")
print("-" * 80)

test_files = list(ROOT.glob("tests/**/*.py"))
example_files = list(ROOT.glob("examples/**/*.py"))

code_lines = sum(len(f.read_text().splitlines()) for f in py_files)
test_lines = sum(len(f.read_text().splitlines()) for f in test_files if '__pycache__' not in str(f))
example_lines = sum(len(f.read_text().splitlines()) for f in example_files if '__pycache__' not in str(f))

print(f"  Core files: {len(py_files)} files, {code_lines:,} lines")
print(f"  Test files: {len(test_files)} files, {test_lines:,} lines")
print(f"  Example files: {len(example_files)} files, {example_lines:,} lines")
print(f"  TOTAL: {code_lines + test_lines + example_lines:,} lines")

# Module breakdown
modules = {}
for py_file in py_files:
    parts = py_file.relative_to(ROOT / "nesy").parts
    if len(parts) > 0:
        module = parts[0]
        modules[module] = modules.get(module, 0) + len(py_file.read_text().splitlines())

print(f"\n  Lines by module:")
for mod, lines in sorted(modules.items(), key=lambda x: x[1], reverse=True):
    print(f"    - {mod}: {lines:,} lines")

# =============================================================================
# 6. GIT STATUS
# =============================================================================
print("\n6. GIT REPOSITORY STATUS")
print("-" * 80)

try:
    result = subprocess.run(['git', 'status', '--short'], capture_output=True, text=True, cwd=ROOT)
    status = result.stdout.strip()
    
    if status:
        print(f"  Modified files:")
        for line in status.split('\n')[:10]:
            print(f"    {line}")
    else:
        print("  ✓ Working directory clean")
    
    # Commits
    result = subprocess.run(['git', 'log', '--oneline', '-5'], capture_output=True, text=True, cwd=ROOT)
    commits = result.stdout.strip().split('\n')
    print(f"\n  Recent commits:")
    for commit in commits:
        print(f"    {commit}")
        
except Exception as e:
    print(f"  ⚠️  Git not available: {e}")

# =============================================================================
# 7. OPTIMIZATION OPPORTUNITIES
# =============================================================================
print("\n7. OPTIMIZATION ANALYSIS")
print("-" * 80)

optimizations = []

for py_file in py_files:
    content = py_file.read_text()
    
    # Check for O(n²) patterns
    if 'for ' in content and content.count('for ') > 1:
        # Nested loops might be O(n²)
        if re.search(r'for\s+\w+.*:\s*\n\s+for\s+\w+', content):
            optimizations.append(f"{py_file.name}: Potential O(n²) nested loop")
    
    # Check for list concatenation in loops
    if '+=' in content and 'for' in content:
        if re.search(r'for.*:\s*\n.*\+=\s*\[', content):
            optimizations.append(f"{py_file.name}: List concatenation in loop (use list comprehension)")

if optimizations:
    print(f"  Found {len(optimizations)} potential optimizations:")
    for opt in optimizations[:5]:
        print(f"    - {opt}")
else:
    print("  ✓ No obvious optimization opportunities detected")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)

issues_found = []
if missing:
    issues_found.append(f"{len(missing)} missing files")
if syntax_errors:
    issues_found.append(f"{len(syntax_errors)} syntax errors")
if sum(len(v) for v in issues.values()) > 10:
    issues_found.append("Multiple code quality issues")

if issues_found:
    print(f"\n⚠️  Issues found: {', '.join(issues_found)}")
else:
    print(f"\n✅ CODE VERIFICATION SUCCESSFUL!")
    print(f"   - {len(py_files)} Python files")
    print(f"   - {code_lines:,} lines of code")
    print(f"   - {len(test_files)} test files")
    print(f"   - All syntax valid")
    print(f"   - Code quality good")

print("\n" + "=" * 80)
