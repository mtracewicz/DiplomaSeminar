#!/bin/bash
pip install -r /DiplomaSeminar/requirements.txt 
groupadd -g $U mtracewicz
useradd -u $U -g $U -ms /bin/bash mtracewicz
chown -R mtracewicz:mtracewicz /DiplomaSeminar
