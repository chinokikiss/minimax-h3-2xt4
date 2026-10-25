import json

with open("kaggle_latest_run.log", "r", encoding="utf-8") as f:
    text = f.read()

lines = text.split("\n")
out = []
for l in lines:
    l = l.strip().lstrip(",").lstrip("[").rstrip("]")
    if '"data":' in l:
        try:
            d = json.loads(l)
            out.append(d.get("data", ""))
        except:
            pass

content = "".join(out)
with open("parsed_kaggle_output.txt", "w", encoding="utf-8") as f:
    f.write(content)

print("Parsed successfully! Total chars:", len(content))
