#!/usr/bin/env python
"""
Fix pickle files compatibility issue with numpy._core
"""
import sys
import os

# Try to import numpy and fix the compatibility issue
try:
    import numpy
    # Add numpy._core compatibility for older pickle files
    import numpy as np
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray
    sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath
except Exception as e:
    print(f"NumPy fix error: {e}")

import joblib
import pickle

def fix_pickle_file(filepath):
    """Load and resave a pickle file to fix compatibility"""
    try:
        print(f"Loading {filepath}...")
        obj = joblib.load(filepath)
        
        print(f"Resaving {filepath}...")
        joblib.dump(obj, filepath, compress=3)
        
        print(f"✅ Fixed: {filepath}")
        return True
    except Exception as e:
        print(f"❌ Error fixing {filepath}: {e}")
        return False

if __name__ == "__main__":
    print("Fixing pickle files compatibility...")
    print("=" * 60)
    
    success = True
    success &= fix_pickle_file('model.pkl')
    success &= fix_pickle_file('scaler.pkl')
    
    print("=" * 60)
    if success:
        print("✅ All pickle files fixed successfully!")
    else:
        print("❌ Some errors occurred")
        sys.exit(1)
