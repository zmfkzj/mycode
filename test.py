import json
from pathlib import Path

with open(str(Path('/Users/minkyu/Desktop/무제 폴더/lv1/result_test.json')),'r') as f:
    a = json.load(f)
b = json.dumps(a,ensure_ascii=False, indent=4)
print(a)