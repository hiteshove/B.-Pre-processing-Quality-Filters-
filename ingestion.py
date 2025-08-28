import os
import pandas as pd

def ingest_files(input_dir: str):
    records = []
    asset_id = 1
    for root, _, files in os.walk(input_dir):
        for file in files:
            ext = file.split(".")[-1].lower()
            if ext in ["jpg", "jpeg", "png"]:
                ftype = "image"
            elif ext in ["pdf", "docx", "txt"]:
                ftype = "document"
            elif ext in ["mp3", "wav"]:
                ftype = "audio"
            elif ext in ["mp4", "avi", "mov"]:
                ftype = "video"
            else:
                continue

            records.append({
                "asset_id": f"{asset_id:03d}",
                "filename": file,
                "type": ftype,
                "quality": None,
                "status": "pending",
                "reason": "-",
                "subtype": "-"
            })
            asset_id += 1

    return pd.DataFrame(records)
