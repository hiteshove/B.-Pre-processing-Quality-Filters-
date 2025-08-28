import pandas as pd
from pipeline import ingestion, corrections, filters, classification, utils

INPUT_DIR = "data/sample/"
OUTPUT_FILE = "data/results.csv"

def run_pipeline():
    df = ingestion.ingest_files(INPUT_DIR)

    for idx, row in df.iterrows():
        fpath = utils.get_file_path(INPUT_DIR, row["filename"])

        if row["type"] == "image":
            img = corrections.correct_image(fpath)
            q, reason = filters.image_quality_check(img)
            df.at[idx, "quality"] = q
            df.at[idx, "status"] = "accepted" if q >= 0.7 else "rejected"
            df.at[idx, "reason"] = reason
            df.at[idx, "subtype"] = classification.classify_image(fpath)

        elif row["type"] == "document":
            pages = corrections.correct_pdf(fpath)
            df.at[idx, "quality"] = 0.9 if pages > 0 else 0.0
            df.at[idx, "status"] = "accepted" if pages > 0 else "rejected"
            df.at[idx, "reason"] = "-" if pages > 0 else "corrupted"
            df.at[idx, "subtype"] = classification.classify_document(fpath)

        elif row["type"] == "audio":
            q, reason = filters.audio_quality_check(fpath)
            df.at[idx, "quality"] = q
            df.at[idx, "status"] = "accepted" if q >= 0.7 else "rejected"
            df.at[idx, "reason"] = reason
            df.at[idx, "subtype"] = classification.classify_audio(fpath)

        elif row["type"] == "video":
            q, reason = filters.video_quality_check(fpath)
            df.at[idx, "quality"] = q
            df.at[idx, "status"] = "accepted" if q >= 0.7 else "rejected"
            df.at[idx, "reason"] = reason
            df.at[idx, "subtype"] = classification.classify_video(fpath)

    df.to_csv(OUTPUT_FILE, index=False)
    print("Pipeline finished. Results saved to", OUTPUT_FILE)

if __name__ == "__main__":
    run_pipeline()
