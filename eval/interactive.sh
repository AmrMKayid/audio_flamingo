# IMAGE=gitlab-master.nvidia.com/zkong/audio_flamingo_v1/audiolm:0.2
IMAGE="/lustre/fsw/portfolios/adlr/users/zkong/docker/audiolm-0.2/image.sqsh"

submit_job -i -n interactive \
    --gpu 1 \
    --duration 2 \
    --image $IMAGE \
    --mounts /home/zkong,/lustre/fsw/portfolios/adlr/users/zkong
