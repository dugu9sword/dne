import json
x = json.load(open("external_data/euc-top8.json"))
for k in x:
    if len(x[k]) != 8:
        print(k)