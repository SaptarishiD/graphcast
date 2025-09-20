import os
import subprocess
import argparse
from ecmwfapi import ECMWFService
import yaml
from datetime import datetime, timedelta
from tqdm import tqdm

server = ECMWFService("mars")

# Argument parser to get the SCP target from command line
parser = argparse.ArgumentParser(description="Download HRES forecast and optionally scp to remote.")

# python download_hres_pipe.py --scp_target saptarishi.dhanuka_asp25@10.1.7.56:/Datastorage/saptarishi.dhanuka_asp25/forecasts_hres/raw_hres/

parser.add_argument("--scp_target", type=str, default=None, help="user@host:/path/to/target")
args = parser.parse_args()
scp_target = args.scp_target

yaml_file_name = "download_hres.yaml"

# Load configuration
with open(yaml_file_name, "r") as f:
    config = yaml.safe_load(f)

for entry in config.get("dates", []):
    # Read block parameters
    year = entry["year"]
    month = entry["month"]
    start_day = entry["start_day"]
    interval = entry["initialisation_interval"]
    num = entry["number_of_initialisations"]

    # Build the starting date
    current = datetime(year, month, start_day)

    for i in tqdm(range(num)):
        # Offset date by i * interval days
        request_date = current + timedelta(days=i * interval)
        if (i == 0 or i == 32):
            print(request_date)
        y, m, d = request_date.year, request_date.month, request_date.day

        date_str = f"{y}{m:02d}{d:02d}"
        output_file = f"/Datastorage/saptarishi.dhanuka_asp25/forecasts_hres/raw_hres/{year}/hres_{y}_{m}_{d}_ppt_6hourly.grib"
        if os.path.exists(output_file):
            print(f"File {output_file} already exists, skipping download.")
            continue


        print(f"Requesting: {date_str} → {output_file}")
        server.execute(
            {
                "class": "od",
                "date": date_str,
                "expver": "1",
                "levtype": "sfc",
                "param": "228.128",
                "step": "0/6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120/126/132/138/144/150/156/162/168",
                "stream": "oper",
                "time": "00:00:00",
                "type": "fc"
            },
            output_file
        )

        # After download, scp the file if target specified
        if scp_target:
            print(f"Transferring {output_file} to {scp_target}")
            try:
                subprocess.run(["scp", output_file, scp_target], check=True)
                os.remove(output_file)
                print(f"Completed download and transfer of {output_file}")
            except subprocess.CalledProcessError as e:
                print(f"Error during SCP: {e}")
            except OSError as e:
                print(f"Error deleting file {output_file}: {e}")
