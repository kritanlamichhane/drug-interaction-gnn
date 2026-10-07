import pandas as pd

cache = pd.read_csv("data/processed/smiles_cache.csv", index_col=0)

print(cache.head())
print(cache["smiles"].notna().sum())


import requests

cid = 2173

url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"

r = requests.get(url)

print("status:", r.status_code)
print(r.text[:500])
