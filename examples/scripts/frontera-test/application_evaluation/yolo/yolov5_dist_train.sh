#!/bin/sh
#Set debug on
#set -x 

# python -m torch.utils.collect_env

source /home1/09168/ldai1/anaconda3/etc/profile.d/conda.sh

source /home1/09168/ldai1/anaconda3/bin/activate yolov5 
export NCCL_DEBUG=INFO

HOSTNAME=`hostname -s`
MASTER_PORT=1234

if [ ! -f /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_2nodes ]
then
  echo "Please create /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_2nodes!"
  exit 1
fi

MASTER=`cat /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_2nodes | head -1`
# Extract the Infiniband IP address and concat with the port address
MASTER_IP=($(cat /etc/hosts | grep "$MASTER" | awk '{ print $1}'))
echo "MASTER will start in host: $MASTER, IP: $MASTER_PORT "

WORKER=() # including master node
WORKER_IP=""
for worker in `cat /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_2nodes`; do
	WORKER+=($worker)
  	temp_ip=($(cat /etc/hosts | grep "$worker-ib" | awk '{ print $1}'))
	WORKER_IP="$WORKER_IP,$temp_ip"
done

WORKER_IP=${WORKER_IP#?}

# Find the index of the worker
get_index() {
	for i in "${!WORKER[@]}"; do
		if [[ "${WORKER[$i]}" = "$1" ]]; then
			echo "${i}";
		fi
	done
}

WORK_INDEX=`get_index $HOSTNAME`

temp=($(wc /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_2nodes))
HOST_NUM=${temp[0]} # Count the number of hosts

python -m torch.distributed.run --nproc_per_node 1 --nnodes $HOST_NUM --node_rank $WORK_INDEX --master_addr $MASTER_IP \
	--master_port $MASTER_PORT /home1/09168/ldai1/yolov5/train.py \
	--epochs 100 --batch 128 --data coco128.yaml --cfg yolov5s.yaml --weights ''

# # Clean up
# for host in `cat /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_2nodes`; do
# 	ssh $host killall -9 python
# 	ssh $host killall -9 mpirun
# done

