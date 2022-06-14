#!/bin/bash
source credentials
input="$1"
echo "Reading file: $1"
while IFS= read -r line
do
  wget -r -N -c -np --user $USER --password $PASS https://physionet.org/files/mimic-cxr/2.0.0/"$line"
done < "$input"
