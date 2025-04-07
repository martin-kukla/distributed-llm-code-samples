#! /bin/bash
apt-get update
apt-get -y install git

git config --global user.email "martin.kukla@cantab.net"
git config --global user.name "Martin Kukla"
git config --global credential.helper store # This is not secure...
git config --global --add safe.directory /efs/notebooks/mkukla/distributed-llm-code-samples