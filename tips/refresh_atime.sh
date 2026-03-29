#!/bin/bash
#
# Update access time (atime) for specific directories in KISTI NEURON.
# This script prevents files from being deleted by the 15-day purge policy.

# List of directories to manage. Update these paths as needed.
TARGET_DIRS=(
  "/scratch/${USER}/chanwcom"
  "/scratch/${USER}/kaggle"
  "/scratch/${USER}/kir"
  "/scratch/${USER}/ky"
  "/scratch/${USER}/kyungseok"
  "/scratch/${USER}/seongwoon"
  "/scratch/${USER}/yanghoon"
  "/scratch/${USER}/yerin"
)

echo "START!!!"

#######################################
# Check for 'ToBeDelete_' prefix and update atime using 'touch'.
# Globals:
#   TARGET_DIRS
# Arguments:
#   None
#######################################
main() {
  echo "--- atime refresh started: $(date) ---"

  for dir in "${TARGET_DIRS[@]}"; do
	echo $dir
    if [[ ! -d "${dir}" ]]; then
      echo "[SKIP] Directory not found: ${dir}"
      continue
    fi

    echo "[CHECK] Processing: ${dir}"

    # Count files already marked for deletion by the system.
    local pending_count
    pending_count=$(find "${dir}" -name "ToBeDelete_*" | wc -l)

    if [[ "${pending_count}" -gt 0 ]]; then
      echo "  ! WARNING: ${pending_count} files have 'ToBeDelete_' prefix."
      echo "  ! Please rename them manually to stop the deletion process."
    fi

    # Update atime only (-a) without creating new files (-c).
    # Using '+' with -exec improves performance for large numbers of files.
    find "${dir}" -type f -exec touch -ac {} +
    
    echo "  > Success: atime updated for files in ${dir}"
  done

  echo "--- All tasks completed ---"
}

# Execute the main function.
main "$@"
