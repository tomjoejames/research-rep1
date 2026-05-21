#!/bin/bash

REGISTRY_FILE="registry/devices.json"

if [ -f ".device_id" ]; then
    echo "Device already registered:"
    cat .device_id
    exit 0
fi

echo "Enter your invite code:"
read INVITE

DEVICE_ID=$(python3 - <<EOF
import json

with open("$REGISTRY_FILE") as f:
    data = json.load(f)

print(data.get("$INVITE", "INVALID"))
EOF
)

if [ "$DEVICE_ID" = "INVALID" ]; then
    echo "Invalid invite code"
    exit 1
fi

echo "$DEVICE_ID" > .device_id

echo "Registered as: $DEVICE_ID"
