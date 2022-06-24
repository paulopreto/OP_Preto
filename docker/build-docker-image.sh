#!/bin/bash

echo $UID > USER_ID.txt

docker build --rm --tag openpose --file Dockerfile .
