# TITAN: Distributed Vector Search System

## Overview

1. Scheduler Node
2. Worker Node
3. Client

## Quick Start

### Pre-requisites

```bash
# Clone the repository
git clone git@github.com:WHU-STAR/TITAN.git
cd TITAN

# Install the protobuf compiler
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# Install the dependencies
go mod tidy

# create data directory
mkdir data
```

Link or download data to your data directory

```bash
# suppose you are in the root directory of project
cd data
ln -s /path/to/your/data data
## or download data
```

See necessary data path in [data_loader.py](./script/data_loader.py)

### Build

```bash
# Generate grpc code
make grpc

# Build engine
make engine

# Build the binary
make build

# Generate the config file
make config

# Generate pybinding for `data_processor_vamana.py`
make pybind
```

### Generate data

```bash
conda create --name TITAN python=3.8
conda activate TITAN
pip install -r requirements.txt
python script/data_processor.py <dataset_name>
ln -s <YOUR_PROJECT_PATH>/<dataset_name>_worker_data <YOUR_PROJECT_PATH>/data/workerdata (for macOS)
ln -s ./data/<dataset_name>_worker_data ./data/workerdata (for Linux)
```

And you will get the data for Node in `/bin`, such as `worker_1.bin` and `bucket_centroids.bin`.

### Change Dataset

```bash
# in project root
cd data
rm workerdata
ln -s <dataset_name>_worker_data_<method> workerdata
```

Or run the script `script/change_data.sh`, this requires the command-line tool [fzf](https://github.com/junegunn/fzf)

```bash
./script/change_data.sh
```

And select the dataset you want to use.

### Run

#### Run manually

1. Start the scheduler

```bash
cd bin && ./scheduler
# or you can run by following script
TITAN_CONFIG_BUILDER=<yourConfigWantedInRunconfig.yaml> ./script/start_scheduler.sh
```

2. Start the workers

Single machine:

```bash
TITAN_CONFIG_BUILDER=<yourConfigWantedInRunconfig.yaml> ./script/start_workers.sh
```
Options:

- Environment variables:
  - `TITAN_CONFIG_BUILDER`: specify the config to use. If omitted, will use options in `default_builder.go`
  - `BASE_PORT`: optional, port of worker0.

3. Start the client(query interface)

```bash
TITAN_CONFIG_BUILDER=<yourConfigWantedInRunconfig.yaml> ./script/start_client.sh -r <runname>
```

Options:

- Environment variable:
  - `TITAN_CONFIG_BUILDER`: same as above.
- Command-line options:
  - `-r`: specify the run name, which will be used to save the query results. Default is `default_run`.

4. Stop the workers

```bash
./script/kill_workers.sh
```

#### Run with script

This requires the command-line tool [tmuxifier](https://github.com/jimeh/tmuxifier) installed

##### Single machine

- Use -c to specify number of cores to run (in total, not per worker).
- Use -s if you want the script to wait until the experiment is complete (tmux session closed). Useful when automating multiple runs.
- Use -r to specify method defined in `runconfig.yaml` file. Default is `default_local`.
- Use -n to specify csv file name. The csv file will be output to `./log` directory.
- Use -p to specify port of the first worker. Default is 50050.
- Use -b to specify the base number of core. This is useful when running multiple sessions at the same time

These two options are optional, default to 16 cores and 3 seconds.

```bash
cd ./script
./run_session.sh -c 32 -r default_local -n deep10m_100k_0thresh

```

##### Distributed

```bash
cd ./script/
./distributed_run_session.sh -r default_distributed -n nameofcsvfile -u worker -c worker -w 16 -p 50050
```

Options:
- `-r`: Run config name, defined in `runconfig.yaml`
- `-n`: Run name, will be used to save the query results 
- `-u`: User name for ssh to worker nodes, default `$USER`
- `-c`: User name for ssh to client node, default `$USER`
- `-w`: Number of workers, default 16
- `-p`: Base port of workers, default 50050
- `-s`: Wait until the experiment is complete (tmux session closed). See above.


##### Terminate runner

Specifically, inside the _client_ pane of the `runner-*` window, you can use the `./script/terminate_runner.sh` command to both kill workers and kill the tmux window.

#### Logrotate

It has been observed that log files can be quite large, especially when running with a large set of parameters. Keep `./script/logrotate.sh` running in the background (i.e. a tmux window) to keep the log files from growing too large.

Run `make logrotate` to generate the configuration file first.
