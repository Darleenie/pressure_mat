#!/usr/bin/env bash

cd torch-hd
python3 setup.py install
cd ..
python3 HD_perm.py         