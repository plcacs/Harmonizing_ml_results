import json
from collections import defaultdict
with open('mypy_results.json', 'r') as f:
    data = json.load(f)
file_pairs = defaultdict(lambda: {'original': None, 'no_types': None})
for filename, attributes in data.items():
    is_compiled = attributes.get('isCompiled', False)
    if filename.endswith('_no_types.py') and filename.count('_no_types') == 1:
        original_name = filename.replace('_no_types', '')
        file_pairs[original_name]['no_types'] = is_compiled
    elif '_no_types' not in filename:
        file_pairs[filename]['original'] = is_compiled
both_true = 0
both_false = 0
original_false_no_types_true = 0
original_true_no_types_false = 0
for pair in file_pairs.values():
    if pair['original'] is not None and pair['no_types'] is not None:
        if pair['original'] and pair['no_types']:
            both_true += 1
        elif not pair['original'] and (not pair['no_types']):
            both_false += 1
        elif not pair['original'] and pair['no_types']:
            original_false_no_types_true += 1
        elif pair['original'] and (not pair['no_types']):
            original_true_no_types_false += 1
print('Both compiled: ', both_true)
print('Both failed compilation: ', both_false)
print('Original failed, no_types compiled: ', original_false_no_types_true)
print('Original compiled, no_types failed: ', original_true_no_types_false)