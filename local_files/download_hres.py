#!/usr/bin/env python
from ecmwfapi import ECMWFService
import yaml
from datetime import datetime, timedelta
from tqdm import tqdm


server = ECMWFService("mars")

yaml_file_name = "download_hres.yaml"

# Load configuration
with open(yaml_file_name, "r") as f:
    config = yaml.safe_load(f)

for entry in config.get("dates", []):
    # Read block parameters
    year  = entry["year"]
    month = entry["month"]
    start_day = entry["start_day"]
    interval  = entry["initialisation_interval"]
    num       = entry["number_of_initialisations"]

    # Build the starting date
    current = datetime(year, month, start_day)

    for i in tqdm(range(num)):
        # On the i-th run, offset by i * interval days
        request_date = current + timedelta(days=i * interval)
        y, m, d = request_date.year, request_date.month, request_date.day

        date_str    = f"{y}{m:02d}{d:02d}"
        output_file = f"forecast_data/hres_{y}_{m}_{d}_ppt_6hourly.grib"

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