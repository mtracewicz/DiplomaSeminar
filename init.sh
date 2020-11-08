#!/bin/bash
pip install -r /DiplomaSeminar/requirements.txt 
groupadd -g $G mtracewicz
useradd -u $U -g $G -ms /bin/bash mtracewicz
chown -R mtracewicz:mtracewicz /DiplomaSeminar
