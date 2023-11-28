#!/bin/bash
set -e

# Google drive folder ID to download the data from.
# If you don't want to use Google drive, then set the next variable to empty and
# just place the zipped data in the $ROOT_FOLDER according to README.
GDRIVE_FOLDER_ID="1P1jKmmklXwH_lB0js50OzSxiyCAJr"
# Path to the client secret json file. If you would like to use google drive but don't have
# the API keys yet, follow the instructions here: https://stackoverflow.com/a/72076913 
# to generate it from Google developer console.
CLIENT_SECRET_JSON="client_secret.json"

TASK_NO=1 # numbers listed in new_downloader.py
HOME="my_home" # used for folder naming purposes
ROOT_FOLDER="$SCRATCH/finetuning_data" # directory to which data will be downloaded
EXPORT_FOLDER="$SCRATCH/finetuning_data_extracted" # directory to which data will be extracted
ENV_NO=1 # used to name environment within "home" folder
GRIPPER_MODEL_PATH="$SCRATCH/hello-robot-data-collection/gripper_model.pth"

# Check if GDRIVE_FOLDER_ID is set.
if [ ! -z "$GDRIVE_FOLDER_ID" ]; then
    # If we already have the root folder, continue on.
    if [ ! -f "$CLIENT_SECRET_JSON" ]; then
        # First, check if we have the client secret JSON file.
        echo "Client secret JSON file not found. Please follow the instructions here: https://stackoverflow.com/a/72076913 to generate it from Google developer console."
        exit 1
    fi
    echo "Downloading data from Google Drive..."
    python gdrive_downloader.py --folder_id $GDRIVE_FOLDER_ID --task_no $TASK_NO --home $HOME --root_folder $ROOT_FOLDER --env_no $ENV_NO
    sleep 5
    echo "Done!"
fi

echo "Unzipping data..."
python unzip_data.py --source_folder $ROOT_FOLDER --export_folder $EXPORT_FOLDER
sleep 5
echo "Done!"

echo "Processing data..."
python process_from_r3ds.py --r3d_paths_file "${EXPORT_FOLDER}/r3d_files.txt" --model_path $GRIPPER_MODEL_PATH
sleep 5
echo "Done!"

echo "Exporting videos..."
python export_vids_ffmpeg.py --r3d_paths_file "${EXPORT_FOLDER}/r3d_files.txt" --num_workers 1 --start_index 0 --end_index -1
sleep 5
echo "Done!"