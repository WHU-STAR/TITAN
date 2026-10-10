window_root $PWD

# Create new window. If no argument is given, window name will be based on
# layout file name.
new_window "runner-$session_id"

# Split window into panes.
split_h 50
split_v 50
run_cmd "export SESSION_ID=$session_id" 1
run_cmd "num_workers=$num_workers VCLUSTER_CONFIG_BUILDER=$VCLUSTER_CONFIG_BUILDER ./start_scheduler.sh" 1
run_cmd "num_workers=$num_workers BASE_PORT=$BASE_PORT VCLUSTER_CONFIG_BUILDER=$VCLUSTER_CONFIG_BUILDER coreCount=$coreCount session_id=$session_id ./start_workers.sh -c $BASE_CORE" 0
run_cmd "export SESSION_ID=$session_id" 2
run_cmd "VCLUSTER_CONFIG_BUILDER=$VCLUSTER_CONFIG_BUILDER ./start_client.sh -r $runName" 2
run_cmd "./terminate_runner.sh $PROMPTSTOP" 2
