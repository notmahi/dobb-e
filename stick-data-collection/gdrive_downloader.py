import argparse
import asyncio
import io
import json
import logging
import os
import time

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
file_path = os.path.dirname(os.path.realpath(__file__))

# Learn how to create one for yourself: https://stackoverflow.com/a/72076913
CLIENT_SECRET_JSON = os.path.join(file_path, "client_secret.json")
assert os.path.exists(CLIENT_SECRET_JSON), (
    f"Client secret file not found at {CLIENT_SECRET_JSON}, "
    "create one following the instructions at https://stackoverflow.com/a/72076913"
)
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

TASK_NAMES = [
    "Door_Opening",  # 1
    "Door_Closing",  # 2
    "Drawer_Opening",  # 3
    "Drawer_Closing",  # 4
    "Handle Grasping",  # 5
    "Pick_And_Place",  # 6
    "Random",  # 7
    "Switch_Button",  # 8
    "Pouring",  # 9
    "Box_Opening",  # 10
    "Unplugging",  # 11
    "Pulling",  # 12
    "Button_Pressing",  # 13
    "Light_Switch",  # 14
    "Air_Fryer_Opening",  # 15
    "Air_Fryer_Closing",  # 16
    "Cushion_Flip",  # 17
    "PT_Pickup",  # 18
    "Chaining",  # 19
    "Toaster",  # 20
]
SEMAPHORE = asyncio.Semaphore(1)


def get_directory_tree_info(service_account_file, folder_id):
    creds = service_account.Credentials.from_service_account_file(
        service_account_file, scopes=SCOPES
    )
    drive_service = build("drive", "v3", credentials=creds)

    def get_file_info(file_id):
        file_metadata = (
            drive_service.files()
            .get(fileId=file_id, fields="id, name, mimeType, modifiedTime")
            .execute()
        )
        file_name = file_metadata["name"]
        file_type = file_metadata["mimeType"]
        file_modified_time = file_metadata["modifiedTime"]
        if file_type != "application/vnd.google-apps.folder":
            return {
                "name": file_name,
                "type": file_type,
                "modified_time": file_modified_time,
                "id": file_id,
                "contents": [],
            }
        else:
            return get_folder_info(file_id)

    def get_folder_info(folder_id):
        folder_metadata = (
            drive_service.files().get(fileId=folder_id, fields="id, name").execute()
        )
        folder_name = folder_metadata["name"]
        folder_info = {
            "name": folder_name,
            "type": "application/vnd.google-apps.folder",
            "id": folder_id,
            "contents": [],
        }
        query = f"'{folder_id}' in parents and trashed = false"
        results = (
            drive_service.files()
            .list(q=query, fields="nextPageToken, files(id)")
            .execute()
        )
        items = results.get("files", [])
        for item in items:
            item_id = item["id"]
            item_info = get_file_info(item_id)
            folder_info["contents"].append(item_info)
        return folder_info

    folder_info = get_folder_info(folder_id)
    return folder_info


async def download_single_file_async(
    file_id, file_path, drive_service, file_name, file_mime_type
):
    async with SEMAPHORE:
        full_path = os.path.join(file_path, file_name)
        if not os.path.exists(full_path):
            os.makedirs(file_path, exist_ok=True)
            try:
                logging.info(
                    f"Starting download on {file_name} ({file_mime_type}) to {file_path}"
                )
                request = drive_service.files().get_media(fileId=file_id)
                fh = io.FileIO(full_path, mode="wb")
                downloader = MediaIoBaseDownload(
                    fh, request, chunksize=100 * 1024 * 1024
                )
                done = False
                while done is False:
                    status, done = await asyncio.to_thread(downloader.next_chunk)
                    logging.info(f"Download {int(status.progress() * 100)}.")

                logging.info(
                    f"Downloaded {file_name} ({file_mime_type}) to {file_path}"
                )
            except HttpError as error:
                logging.error(
                    f"An error occurred: {error} for file {file_name} ({file_id})"
                )
                os.remove(file_path)
            except OverflowError as error:
                logging.error(
                    f"An error occurred: {error} for file {file_name} ({file_id})"
                )
                os.remove(file_path)

        logging.info(f"Downloaded {file_name} at path {file_path}")
        return file_id, file_path


async def download_drive_folder_async(
    tree_json_file, download_folder_name, client_secret_path=CLIENT_SECRET_JSON
):
    if not os.path.exists(client_secret_path):
        raise FileNotFoundError(f"Client secret file not found at {client_secret_path}")

    if not os.path.exists(download_folder_name):
        os.makedirs(download_folder_name)
        # raise FileNotFoundError(f"Download folder not found at {download_folder_name}")

    # Authenticate with the Google Drive API
    creds = service_account.Credentials.from_service_account_file(client_secret_path)
    drive_service = build("drive", "v3", credentials=creds)

    # Retrieve a list of all files in the shared folder
    with open(tree_json_file, "r") as f:
        tree = json.load(f)

    # Now recursirvely go over the tree and download the files.
    await download_folder_recursively(
        drive_service, download_folder_name, tree, current_drive_path=()
    )


async def download_folder_recursively(
    drive_service,
    download_path,
    current_tree,
    current_drive_path=(),
):
    if (
        len(current_tree["contents"]) == 0
        and current_tree["type"] != "application/vnd.google-apps.folder"
    ):
        # We have reached individual files, download them.
        await download_single_file_async(
            file_id=current_tree["id"],
            file_path=os.path.join(download_path, *current_drive_path[:-1]),
            drive_service=drive_service,
            file_name=current_tree["name"],
            file_mime_type=current_tree["type"],
        )
    else:
        # Recursively download the files.
        await asyncio.gather(
            *[
                download_folder_recursively(
                    drive_service,
                    download_path,
                    item,
                    current_drive_path + (item["name"],),
                )
                for item in current_tree["contents"]
            ]
        )


parser = argparse.ArgumentParser()
parser.add_argument(
    "--folder_id",
    type=str,
    required=True,
    help="Folder ID of the shared Google drive folder",
)
parser.add_argument(
    "--task_no",
    type=int,
    required=True,
    help="Task category identifier number",
    default=-1,
)
parser.add_argument(
    "--task_name",
    type=str,
    required=False,
    help="Task category name",
)
parser.add_argument(
    "--home",
    type=str,
    required=True,
    help="Home number or name",
)
parser.add_argument(
    "--root_folder",
    type=str,
    required=True,
    help="Folder to export the data to",
)
parser.add_argument(
    "--env_no",
    type=int,
    required=True,
    help="Current env number within this home and task category",
)

args = parser.parse_args()
params = vars(args)

if __name__ == "__main__":
    folder_id = params["folder_id"]
    task_no = params["task_no"]
    home = params["home"]
    root_folder = params["root_folder"]
    env = "Env" + str(params["env_no"])

    if params["task_name"] is not None:
        task = params["task_name"]
    else:
        assert 0 < task_no <= len(TASK_NAMES)
        task = TASK_NAMES[task_no - 1]

    tree_info = get_directory_tree_info(CLIENT_SECRET_JSON, folder_id)
    os.makedirs(root_folder, exist_ok=True)
    with open(os.path.join(root_folder, f"tree_info.json"), "w") as f:
        json.dump(tree_info, f, indent=4)

    time.sleep(2)

    download_folder_name = os.path.join(root_folder, task, home, env)
    tree_json_file = os.path.join(root_folder, "tree_info.json")

    # Asynchronously download the files.
    asyncio.get_event_loop().run_until_complete(
        download_drive_folder_async(tree_json_file, download_folder_name)
    )

    os.remove(tree_json_file)
