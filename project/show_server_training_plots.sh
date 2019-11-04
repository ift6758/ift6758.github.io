#!/bin/bash
# scp -r user07@35.208.187.181:submissions/checkpoints ./server_checkpoints
# scp -r user07@35.208.187.181:submissions/logs ./server_logs
rsync -a --update --verbose user07@35.208.187.181:submissions/checkpoints/ :submissions/logs server_checkpoints/
cp -r server_checkpoints/logs/ server_logs/
tensorboard --logdir server_checkpoints