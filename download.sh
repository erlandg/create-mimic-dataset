#!/bin/bash
input="$1"
echo "Reading file: $1"
while IFS= read -r line
do
  wget -r -N -c -np --user INSERT_USERNAME_HERE --password INSERT_PASSWORD_HERE https://physionet.org/files/mimic-cxr/2.0.0/"$line"
done < "$input"
