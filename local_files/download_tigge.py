#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer

server = ECMWFDataServer()

server.retrieve({
    "class": "ti",
    "dataset": "tigge",
    "date": "2020-08-01/to/2020-08-31",
    "expver": "prod",
    "grid": "0.5/0.5",
    "levtype": "sfc",
    "origin": "vabb",
    "param": "167/228228",
    "step": "0/6/12/18/24/30/36/42/48",
    "time": "00:00:00",
    "type": "cf",
    "target": "output"
})