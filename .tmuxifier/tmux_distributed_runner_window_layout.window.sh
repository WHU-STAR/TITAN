window_root $PWD

# Create new window. If no argument is given, window name will be based on
# layout file name.
new_window "runner"

# Split window into panes.
split_h 50
split_v 50
run_cmd "export SESSION_ID=$session_id" 1
run_cmd "num_workers=$num_workers TITAN_CONFIG_BUILDER=$TITAN_CONFIG_BUILDER ./start_scheduler.sh" 1
run_cmd "BASE_PORT=$BASE_PORT target_user=$target_user TITAN_CONFIG_BUILDER=$TITAN_CONFIG_BUILDER session_id=$session_id num_workers=$num_workers ./distributed_start_workers.sh" 0
run_cmd "export SESSION_ID=$session_id" 2
# run_cmd "TITAN_CONFIG_BUILDER=$TITAN_CONFIG_BUILDER ./distributed_client.sh -r $runName -u $target_user; echo done" 2
run_cmd "TITAN_CONFIG_BUILDER=$TITAN_CONFIG_BUILDER ./start_client.sh -r $runName" 2
run_cmd "./distributed_terminate_runner.sh $PROMPTSTOP -u $target_user -w $num_workers" 2
