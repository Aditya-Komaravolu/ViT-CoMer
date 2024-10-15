CONFIG=$1
GPUS=2
WORKDIR=$2

nnodes=${NNODES:-1}
PORT=${PORT:-29500}
master_addr=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 \
 -m torch.distributed.launch --nproc_per_node=${GPUS} --nnodes=${nnodes} --node_rank=0  --master_port=$PORT  --master_addr=${master_addr} \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:4} --work-dir $WORKDIR
