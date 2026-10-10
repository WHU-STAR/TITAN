TITAN_CONFIG_BUILDER=default_local
coreCount=16

runName="default_run"

# 自动分配唯一 session id
session_id=0
while [ -d "../pids/$session_id" ]; do
  session_id=$((session_id + 1))
done

BASE_PORT=50050
BASE_CORE=0
num_workers=16

PROMPTSTOP="-p"
do_wait=1 # 0表示等待窗口关闭，1表示不等待
# 如果设成0，tmux窗口在运行完client后会自动关闭。否则不会。目前不支持其他逻辑。

# 使用 getopts 解析选项
while getopts "c:r:n:p:b:w:s" opt; do
  case $opt in
  c)
    coreCount="$OPTARG"
    echo "Corecount 设置为 $coreCount"
    ;;
  r)
    TITAN_CONFIG_BUILDER="$OPTARG"
    echo "TITAN_CONFIG_BUILDER 设置为 $TITAN_CONFIG_BUILDER"
    ;;
  n)
    runName="$OPTARG"
    echo "Run name 设置为 $runName"
    ;;
  p)
    BASE_PORT="$OPTARG"
    echo "Base port 设置为 $BASE_PORT"
    ;;
  b)
    BASE_CORE="$OPTARG"
    echo "Base core 设置为 $BASE_CORE"
    ;;
  w)
    num_workers="$OPTARG"
    echo "Worker 数量设置为 $num_workers"
    ;;
  s)
    do_wait=0
    PROMPTSTOP=""
    echo "设置为运行结束后等待tmux窗口关闭"
    ;;
  \?)
    echo "无效参数: -$OPTARG" >&2
    ;;
  esac
done

PROMPTSTOP=$PROMPTSTOP \
  do_wait=$do_wait \
  num_workers=$num_workers \
  runName=$runName \
  BASE_CORE=$BASE_CORE \
  BASE_PORT=$BASE_PORT \
  TITAN_CONFIG_BUILDER=$TITAN_CONFIG_BUILDER \
  coreCount=$coreCount \
  session_id=$session_id \
  TMUXIFIER_LAYOUT_PATH=$(pwd)/../.tmuxifier \
  tmuxifier load-window tmux_runner_window_layout

echo "do_wait: $do_wait"
if [ $do_wait -eq 0 ]; then
  echo "等待tmux窗口关闭..."
  window_name="runner-$session_id"
  while tmux list-windows -F '#W' 2>/dev/null | grep -q "^${window_name}\$"; do
    sleep 1
  done
  echo "tmux 窗口已关闭，实验完成"
fi
