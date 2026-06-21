import base64
import zlib
import urllib.request
import os

mermaid_code = """flowchart TB
  subgraph Preprocessing ["Data Preprocessing"]
    D["Raw Battery CSVs"] --> L["Sort & Isolate Target"]
    L --> F["Fit Scaler (Train Set)"]
    F --> G["Transform Features"]
    G --> SW["Extract Sliding Windows"]
  end
  subgraph Modeling ["Base Model"]
    SW --> CR["Pretrained Chronos"]
    CR --> CA["Internal Calibration"]
  end
  subgraph Generative ["Generative & SLM"]
    CA --> RE["Train Residual Networks"]
    RE --> SLM["SLM Transformer"]
  end
  subgraph Validation ["Validation Output"]
    SLM --> HY["Confidence Gating"]
    HY --> PRED["Next-Step Prediction"]
  end
  Preprocessing --> Modeling
  Modeling --> Generative
  Generative --> Validation
"""

compressed = zlib.compress(mermaid_code.encode('utf-8'), 9)
b64_encoded = base64.urlsafe_b64encode(compressed).decode('utf-8')

url = f"https://kroki.io/mermaid/png/{b64_encoded}"
output_path = os.path.join("plots", "architecture_diagram.png")
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
    out_file.write(response.read())
print("Saved to", output_path)
