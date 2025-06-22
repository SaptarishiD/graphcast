#!/usr/bin/env python
from ecmwfapi import ECMWFService
import ssl
import urllib3

# Solution 1: Disable SSL verification (use with caution)
# This is the quickest fix but less secure
# ssl._create_default_https_context = ssl._create_unverified_context

# Solution 2: Disable urllib3 warnings (optional, to suppress warnings)
# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Alternative Solution 3: Set SSL context with custom verification
# Uncomment the lines below if you prefer this approach instead of Solution 1
# ssl_context = ssl.create_default_context()
# ssl_context.check_hostname = False
# ssl_context.verify_mode = ssl.CERT_NONE

server = ECMWFService("mars")
server.execute(
    {
        "class": "od",
        "date": "20220701",
        "expver": "1",
        "levtype": "sfc",
        "param": "167.128",
        "step": "0/1/2/3/4/5/6/7",
        "stream": "oper",
        "time": "00:00:00",
        "type": "fc"
    },
    "target.grib")