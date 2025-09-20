from ecmwf.opendata import Client

client = Client(source="ecmwf")

request = {
    "date": "2014-08-10",     # initial forecast date (UTC)
    "time": 0,                # run time: one of 0, 6, 12, or 18
    "type": "fc",             # forecast type
    # specify HRES stream; omit if inferable or use infer_stream_keyword
    "stream": "oper",
    "step": [0,6,12,18,24,30,36,42,48,54,60,66,72,78,84,90,96,102,108,114,120,126,132,138,144,150,156,162,168],     # forecast lead times: 24h, 48h, and 72h
    "param": ["tp"],   # variables: 2 m temperature and mean sea-level pressure
}

# Download into one GRIB file containing fields for all requested steps
result = client.retrieve(request, target="hres_tp20140810.grib2")

print("Downloaded run initialized at:", result.datetime)
