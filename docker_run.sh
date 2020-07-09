PWD=`pwd`
DOCKER_DIR=/`basename $PWD`
CTNRNAME=GpuContainer
function usage()
{
    echo "Usage:"
    echo ""
    echo "./docker_run.sh"
    echo "\t-h --help"
    echo "\t--vendor=$VENDOR (amd or nvidia)"
    echo "\t--mode=$MODE (tf1 or tf2)"
    echo ""
}

while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    VALUE=`echo $1 | awk -F= '{print $2}'`
    case $PARAM in
        -h | --help)
            usage
            exit
            ;;
        --vendor)
            VENDOR=$VALUE
            ;;
        --mode)
            MODE=$VALUE
            ;;
        *)
            echo "ERROR: unknown parameter \"$PARAM\""
            usage
            exit 1
            ;;
    esac
    shift
done

if [ -z "$VENDOR" ]
then
      VENDOR="amd"
      echo " SET VENDOR=$VENDOR"
fi

if [ -z "$MODE" ]
then
      MODE="tf2"
      echo " SET MODE=$MODE"
fi

if [ "$VENDOR" = "amd" ]
then
    if [ "$MODE" = "tf1" ]
    then
        IMAGE="rocm/tensorflow:rocm3.3-tf1.15-dev"
    else
#        IMAGE="rocm/tensorflow:rocm3.3-tf2.1-dev"
        IMAGE="ekuznetsov139/bert_tf2:200616"
    fi
fi

if [ "$VENDOR" = "nvidia" ]
then
    if [ "$MODE" = "tf1" ]
    then
        IMAGE="nvcr.io/nvidia/tensorflow:20.03-tf1-py3"
    else
        IMAGE="nvcr.io/nvidia/tensorflow:20.06-tf2-py3"
    fi
fi

# -u `id -u`:`id -g`
if [ "$VENDOR" = "amd" ]
then
    docker run -it --name $CTNRNAME --network=host  --ipc=host --shm-size 16G  -v=`pwd`:$DOCKER_DIR \
-v /data:/data -w $DOCKER_DIR --privileged --rm --device=/dev/kfd --device=/dev/dri --group-add video \
--cap-add=SYS_PTRACE --security-opt seccomp=unconfined $IMAGE
else
    docker run -it --name $CTNRNAME --gpus 1 --network=host --shm-size=16g  --ulimit memlock=-1 --ulimit stack=67108864 \
-v=`pwd`:$DOCKER_DIR -v /data:/data -w $DOCKER_DIR --privileged --rm --device=/dev/kfd --device=/dev/dri \
--group-add video  --security-opt seccomp=unconfined $IMAGE 
fi

