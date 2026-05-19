import os
import glob
import pandas as pd
import ast
import re
import sys

base = "c:/Adhi/Startup/Research/research-rep1-v2"

print("==============================")
print("PASS 1: Data Integrity (CSVs)")
print("==============================")
p1_anomalies = []
res_dir = os.path.join(base, "results")
csv_files = [f for f in glob.glob(os.path.join(res_dir, "*.csv"))]

exp_headers = {}
for f in csv_files:
    filename = os.path.basename(f)
    try:
        df = pd.read_csv(f)
    except Exception as e:
        p1_anomalies.append(f"{filename}: Failed to read ({e})")
        continue
        
    if df.empty:
        p1_anomalies.append(f"{filename}: File is empty.")
    if df.isna().any().any():
        p1_anomalies.append(f"{filename}: Contains NaNs in critical columns.")
    if 'device' not in df.columns:
        p1_anomalies.append(f"{filename}: Missing 'device' column.")
    else:
        invalid_devices = df[~df['device'].astype(str).str.match(r'^device\d+$')]['device'].unique()
        if len(invalid_devices) > 0:
            p1_anomalies.append(f"{filename}: Invalid device identifiers found -> {invalid_devices}")
            
    exp_prefix = filename.split('_')[0]
    headers = tuple(df.columns)
    if exp_prefix not in exp_headers:
        exp_headers[exp_prefix] = headers
    else:
        if exp_headers[exp_prefix] != headers:
            p1_anomalies.append(f"{filename}: Column header mismatch within {exp_prefix} group.")

if not p1_anomalies:
    print("PASS: Zero Defects")
else:
    for a in p1_anomalies: print(f" - {a}")


print("\n==============================")
print("PASS 2: Code Health & Cross-References")
print("==============================")
p2_anomalies = []

reqs = []
if os.path.exists(os.path.join(base, "requirements.txt")):
    with open(os.path.join(base, "requirements.txt")) as f:
        reqs = [line.split('>=')[0].split('==')[0].strip().lower() for line in f if line.strip()]

class Analyzer(ast.NodeVisitor):
    def __init__(self):
        self.imports = {}
        self.used_names = set()
        self.abs_paths = []
    
    def visit_Import(self, node):
        for n in node.names:
            alias = n.asname if n.asname else n.name.split('.')[0]
            self.imports[alias] = n.name.split('.')[0]
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        if node.module:
            for n in node.names:
                alias = n.asname if n.asname else n.name
                self.imports[alias] = node.module.split('.')[0]
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
        self.generic_visit(node)
        
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            if re.match(r'^[a-zA-Z]:\\', node.value) or node.value.startswith('/Users/') or node.value.startswith('/home/'):
                self.abs_paths.append(node.value)
        self.generic_visit(node)

py_files = glob.glob(os.path.join(base, "experiments", "*.py")) + [os.path.join(base, "plot_results.py")]
all_external_imports = set()
standard_libs = set(sys.builtin_module_names).union({
    'os', 'sys', 'time', 'json', 're', 'importlib', 'glob', 'math', 'csv', 'threading', 'subprocess', 'ast', 'datetime'
})

for py in py_files:
    if not os.path.exists(py): continue
    filename = os.path.basename(py)
    with open(py, 'r', encoding='utf-8') as f:
        try:
            tree = ast.parse(f.read(), filename=py)
            analyzer = Analyzer()
            analyzer.visit(tree)
            
            # Check unused imports
            unused = set(analyzer.imports.keys()) - analyzer.used_names
            # Exclude wildcard or dynamic imports that might be false positives
            unused = {u for u in unused if u != '*'}
            if unused:
                p2_anomalies.append(f"{filename}: Unused imports detected -> {unused}")
                
            # Check hardcoded paths
            if analyzer.abs_paths:
                p2_anomalies.append(f"{filename}: Hardcoded absolute paths found -> {analyzer.abs_paths[:2]}")
                
            # Track external imports
            for alias, pkg in analyzer.imports.items():
                if pkg.lower() not in standard_libs and pkg != 'lib' and pkg != 'experiments':
                    all_external_imports.add(pkg.lower())
                    
        except Exception as e:
            p2_anomalies.append(f"{filename}: AST parse error ({e})")

for imp in all_external_imports:
    if imp not in reqs:
        p2_anomalies.append(f"Global: External library '{imp}' imported but missing from requirements.txt")

if not p2_anomalies:
    print("PASS: Zero Defects")
else:
    for a in p2_anomalies: print(f" - {a}")


print("\n==============================")
print("PASS 3: Reproducibility & Storefront")
print("==============================")
p3_anomalies = []
readme = os.path.join(base, "README.md")
with open(readme, 'r', encoding='utf-8') as f:
    readme_content = f.read()

# Verify explicit paths mentioned in README file tree
# Extracting tree structure lines
tree_lines = re.findall(r'├── (.*?)\s+#', readme_content) + re.findall(r'└── (.*?)\s+#', readme_content)
for t in tree_lines:
    path = t.strip()
    if not path or '*' in path or '<' in path: continue
    
    # Reconstruct relative paths based on indentation
    # For simplicity, just search if the file exists *anywhere* in the repo or exactly at root
    if path.endswith('/'):
        # It's a directory
        dir_path = os.path.join(base, path)
        if not os.path.isdir(dir_path) and not glob.glob(os.path.join(base, "**", path), recursive=True):
            p3_anomalies.append(f"README Tree: Directory '{path}' does not exist.")
    else:
        file_path = os.path.join(base, path)
        if not os.path.exists(file_path):
            # Check if it exists in a subfolder mentioned (like .agents/skills/skill.py)
            if not glob.glob(os.path.join(base, "**", path), recursive=True):
                if path != 'skill.py' and path != 'SKILL.md': # Special cases for MVP pattern
                    p3_anomalies.append(f"README Tree: File '{path}' does not exist.")

# Extract Quick Start bash commands
quick_start_section = re.search(r'## Quick Start(.*?)##', readme_content, re.DOTALL)
if quick_start_section:
    qs_text = quick_start_section.group(1)
    commands = re.findall(r'^(?:python|\./|bash|chmod \+x)\s+([^\s]+)', qs_text, re.MULTILINE)
    for cmd in commands:
        if cmd == 'ollama': continue # Skip external binaries
        cmd_path = os.path.join(base, cmd)
        if not os.path.exists(cmd_path) and '*' not in cmd:
            p3_anomalies.append(f"Quick Start: Target script missing -> {cmd}")

if not p3_anomalies:
    print("PASS: Zero Defects")
else:
    for a in p3_anomalies: print(f" - {a}")
