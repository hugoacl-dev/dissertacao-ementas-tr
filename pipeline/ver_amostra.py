import json
import re

with open('dados_limpos.json') as f:
    d = json.load(f)

for item in d[50:52]:
    print("ID:", item['id'])
    print(item['fundamentacao'][:400] + "\n...\n" + item['fundamentacao'][-200:])
    print("-" * 50)
