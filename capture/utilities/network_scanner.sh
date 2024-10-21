#!/bin/bash

INTERFACE=$(ip route | grep '^default' | head -n 1 | awk '{print $5}')

IPADDR=$(ip -o -4 addr show $INTERFACE | awk '{print $4}')

NETWORK_ADDR=$(echo $IPADDR | awk -F/ '{print $1}' | awk -F. '{print $1"."$2"."$3".0/24"}')

resultFile="../config/cameras.txt"
Port=554

nmap -sT $NETWORK_ADDR -p $Port -oG Scan --open

awk '/open/ { print $2 }' Scan > $resultFile

cat $resultFile