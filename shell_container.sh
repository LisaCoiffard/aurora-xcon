#!/usr/bin/env bash

# Parse arguments
OPTS=$(getopt -o w -l writable -- "$@")
if [ $? != 0 ]; then echo "Failed to parse arguments." >&2 ; exit 1 ; fi
eval set -- "$OPTS"

# Extract options
writable_flag=""
while true ; do
	case "$1" in
		-w | --writable )
			writable_flag="--writable"; shift ;;
		-- )
			shift; break ;;
		* )
			break ;;
	esac
done

# Shell into the container
apptainer shell \
	--bind $(pwd):/workspace/src/ \
	--cleanenv \
	--containall \
	--home /tmp/ \
	--no-home \
	--nv \
	--pwd /workspace/src/ \
	--workdir apptainer/ \
	$writable_flag \
	apptainer/container.sif
