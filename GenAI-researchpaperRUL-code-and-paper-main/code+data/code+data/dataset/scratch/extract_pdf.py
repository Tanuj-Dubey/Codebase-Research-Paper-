import PyPDF2
import sys
import re

pdf_path = r"c:\Users\AADITYA COM\OneDrive\Desktop\GenAI-researchpaperRUL-code-and-paper-main\papers\papers\basepaper -3 AUTOML-Shivendu -Super Computing.pdf"

try:
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        print(f"Total pages: {len(reader.pages)}")
        
        # Search for metrics near B0005 or B05
        for i in range(len(reader.pages)):
            text = reader.pages[i].extract_text()
            if text:
                if 'B0005' in text or 'B05' in text or 'B5' in text or 'RMSE' in text or 'MAE' in text:
                    # Look for tables or metric sections
                    lines = text.split('\n')
                    for j, line in enumerate(lines):
                        if 'B0005' in line or 'B05' in line or 'RMSE' in line or 'MAE' in line:
                            # Print context
                            start = max(0, j-2)
                            end = min(len(lines), j+3)
                            print(f"--- Page {i+1} Context ---")
                            for k in range(start, end):
                                print(lines[k].encode('ascii', 'ignore').decode())
except Exception as e:
    print(f"Error reading PDF: {e}")
