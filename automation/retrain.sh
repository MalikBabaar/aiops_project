#!/bin/bash
# Calls backend retrain API
curl -X POST http://backend:5000/retrain \
     -H "Content-Type: application/json"
