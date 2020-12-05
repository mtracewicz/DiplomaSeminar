#!/bin/bash
docker run --gpus all -it --rm -e U=$UID --name U-NET -v $(pwd):/DiplomaSeminar tensorflow/tensorflow:latest-gpu bash -c -c "/DiplomaSeminar/010_Docker/init.sh;su ds"
