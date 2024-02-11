import os
import glob

# List all Python files in the current directory
py_files = glob.glob(os.path.dirname(__file__) + "/*.py")

# Import all classes from Python files
for py_file in py_files:
    if not py_file.endswith('__init__.py'):
        module_name = os.path.basename(py_file)[:-3]
        exec(f"from .{module_name} import *")