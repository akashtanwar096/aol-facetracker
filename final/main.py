import os
import argparse
from final.get_report_s3 import get_report_optimized, face_report
# from db import setup_database
# from process_event_folder import process_event_folder
# from cluster_faces import cluster_faces
# from cluster_faces import save_cluster_cutouts

def main(start_date, end_date, cli=True):
    """Main function to show the report."""

    print(f"\n📊 Generating report from {start_date} to {end_date}...\n")
    report = get_report_optimized(start_date, end_date)

    if not cli:
        return report

    print("📋 Report:")
    print("--------------------------------------------------")
    print("Cluster ID | Count | Cutout Image | Example Image")
    print("--------------------------------------------------")
    print(report)

    # for cluster_id, data in report.items():
    #     print(f"🟢 Cluster {cluster_id}: Count = {data['count']}")
    #     print(f"   Cutout Image: {data['cutout']}")
    #     print(f"   Example Image: {data['image']}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and show face recognition report.")
    parser.add_argument("start_date", type=str, help="Start date for the report in YYYY-MM-DD format")
    parser.add_argument("end_date", type=str, help="End date for the report in YYYY-MM-DD format")
    args = parser.parse_args()

    main(args.start_date, args.end_date)
