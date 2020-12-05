#!/bin/bash
/usr/bin/python3 -m pip install --upgrade pip
pip install -r /DiplomaSeminar/requirements.txt 
groupadd -g $U ds
useradd -u $U -g $U -ms /bin/bash ds
chown -R ds:ds /DiplomaSeminar
