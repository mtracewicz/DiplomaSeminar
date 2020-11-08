#!/bin/bash
docker run --gpus all -it --rm -e U=$UID --name U-NET -v /home/mtracewicz/DiplomaSeminar/:/DiplomaSeminar tensorflow/tensorflow:latest-gpu bash -c -c "/DiplomaSeminar/init.sh;su mtracewicz"
