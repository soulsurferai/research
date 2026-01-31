"""
json_utils.py - JSON encoding utilities
"""

import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types"""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, (list, tuple)) and len(obj) > 0:
            # Handle lists/tuples of numpy types
            if isinstance(obj[0], (np.integer, np.floating)):
                return [self.default(x) for x in obj]
        return super(NumpyEncoder, self).default(obj)


def save_json(data, filepath, indent=2):
    """Save data to JSON file with numpy type handling"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, cls=NumpyEncoder)


def load_json(filepath):
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)
