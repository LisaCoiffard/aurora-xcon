#!/usr/bin/env bash

# Parse arguments
OPTS=$(getopt -o s -l sandbox -- "$@")
if [ $? != 0 ]; then echo "Failed to parse arguments." >&2; exit 1; fi
eval set -- "$OPTS"

# Extract options
sandbox_flag=""
while true; do
	case "$1" in
		-s | --sandbox )
			sandbox_flag="--sandbox"; shift;;
		-- )
			shift; break;;
		* )
			break;;
	esac
done

if [ $# -gt 0 ]; then  # Last argument is commit, if any
	timestamp=$(date +"%Y-%m-%d_%H%M%S")
	commit=$1
	container_name="container_${timestamp}_${commit}.sif"
else  # Else, default to HEAD
	commit=$(git rev-parse HEAD)
	container_name="container.sif"
fi

# Determine the file system type
file_system=$(stat -f -L -c %T .)

# Build the container
if [[ $file_system == "nfs" && $sandbox_flag == "--sandbox" ]]; then
	# Build container in /tmp/ and then copy back to apptainer/
	tmp_dir=$(mktemp -d)
	apptainer build \
		--build-arg commit=$commit \
		--build-arg github_user=$GITHUB_USER \
		--build-arg github_token=$GITHUB_TOKEN \
		--build-arg gitlab_user=$GITLAB_USER \
		--build-arg gitlab_token=$GITLAB_TOKEN \
        --build-arg wandb_api_key=$WANDB_API_KEY \
        --build-arg cuda_devices=$CUDA_VISIBLE_DEVICES \
		--fakeroot \
		--force \
		$sandbox_flag \
		--warn-unused-build-args \
		$tmp_dir/$container_name apptainer/container.def
	cp -r $tmp_dir/$container_name apptainer/
	rm -rf $tmp_dir
else
	# Build container in apptainer/ directly
	apptainer build \
		--build-arg commit=$commit \
		--build-arg github_user=$GITHUB_USER \
		--build-arg github_token=$GITHUB_TOKEN \
		--build-arg gitlab_user=$GITLAB_USER \
		--build-arg gitlab_token=$GITLAB_TOKEN \
        --build-arg wandb_api_key=$WANDB_API_KEY \
        --build-arg cuda_devices=$CUDA_VISIBLE_DEVICES \
		--fakeroot \
		--force \
		$sandbox_flag \
		--warn-unused-build-args \
		apptainer/$container_name apptainer/container.def
fi
