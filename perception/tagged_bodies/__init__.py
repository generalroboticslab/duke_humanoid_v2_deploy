"""Auto-collect all body configs from type subdirectories."""
import importlib
import pkgutil

ALL_CONFIGS = []

# Iterate over all modules/packages in the current directory (`__path__`)
for mod_info in pkgutil.iter_modules(__path__):
    # Dynamically import the module (e.g., `import tagged_bodies.chamfered_cube`)
    mod = importlib.import_module(f".{mod_info.name}", __name__)
    
    # If the imported module exposes `ALL_CONFIGS`, add them to our global list
    if hasattr(mod, 'ALL_CONFIGS'):
        ALL_CONFIGS.extend(mod.ALL_CONFIGS)
