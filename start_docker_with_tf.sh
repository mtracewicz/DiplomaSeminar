#!/bin/bash
docker run --gpus all -it --rm -e U=$UID -e G=$GID --name U-NET -v /home/mtracewicz/DiplomaSeminar/:/DiplomaSeminar tensorflow/tensorflow:latest-gpu bash
