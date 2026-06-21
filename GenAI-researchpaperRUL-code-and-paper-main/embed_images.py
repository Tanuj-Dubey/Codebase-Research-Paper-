import base64
import os
import re

HTML_PATH = "Final_Research_Report.html"
OUTPUT_PATH = "Shareable_Research_Report.html"
PLOTS_DIR = "plots"

with open(HTML_PATH, "r", encoding="utf-8") as f:
    html_content = f.read()

# Find all <img src="plots/filename.png" ...>
img_tags = re.findall(r'<img [^>]*src="plots/([^"]+)"[^>]*>', html_content)

for img_name in img_tags:
    img_path = os.path.join(PLOTS_DIR, img_name)
    if os.path.exists(img_path):
        print(f"Embedding {img_name}...")
        with open(img_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            mime_type = "image/png"
            base64_src = f"data:{mime_type};base64,{encoded_string}"
            # Replace the specific src attribute
            html_content = html_content.replace(f'src="plots/{img_name}"', f'src="{base64_src}"')

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Standalone report created: {OUTPUT_PATH}")
print("You can now share this single file anywhere!")
