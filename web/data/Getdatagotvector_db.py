import pandas as pd
import json
import numpy as np
datas = np.array(pd.read_csv("../../data/knowledgeBases/base.csv")[["text","embedding"]]).tolist()
output = []
for data in datas:
    output.append({
        "vector": "[" + ",".join([str(i) for i in json.loads(data[1])[:10]]) + "]",
        "text": data[0]
    })
with open("./vector_data.json", "w") as f:
    f.write(json.dumps(output))
