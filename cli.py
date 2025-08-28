import argparse
from pathlib import Path
from .pipeline import DataIngestionPipeline

def main():
    ap = argparse.ArgumentParser(description="Gen AI Data Ingestion: preprocess, filter, classify assets.")
    ap.add_argument("input_dir", help="Path to input assets directory")
    ap.add_argument("-o", "--output-dir", default="processed_assets", help="Directory to write outputs")
    args = ap.parse_args()

    inp = Path(args.input_dir)
    if not inp.exists():
        raise SystemExit(f"Input directory not found: {inp}")

    pipeline = DataIngestionPipeline(str(inp), args.output_dir)
    df = pipeline.run()
    # Print a compact summary row count
    accepted = (df["status"] == "accepted").sum()
    rejected = (df["status"] == "rejected").sum()
    print(f"\nDone. Accepted={accepted}  Rejected={rejected}  Report={Path(args.output_dir) / 'ingestion_report.csv'}")

if __name__ == "__main__":
    main()
