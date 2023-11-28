import argparse
import glob
import json
import logging
import os
import shutil
import zipfile

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--source_folder",
    type=str,
    required=True,
    help="Folder ID of the shared drive folder",
)
parser.add_argument(
    "--export_folder",
    type=str,
    required=True,
    help="Folder to export the data to",
)

args = parser.parse_args()
params = vars(args)

# Set the folder ID of the shared folder
export_folder = params["export_folder"]


def compare_directory_structure(zip_file_path, folder_path, exclude_patterns=None):
    if exclude_patterns is None:
        exclude_patterns = []

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_contents = set(zip_ref.namelist())

    # Exclude files and directories based on the exclusion patterns
    for pattern in exclude_patterns:
        zip_contents = {item for item in zip_contents if pattern not in item}

    folder_contents = set()
    for root, dirs, files in os.walk(folder_path):
        if root != folder_path:
            relative_path = os.path.relpath(root, folder_path)
            folder_contents.add(relative_path + "/")
            for file_name in files:
                folder_contents.add(os.path.join(relative_path, file_name))

    # Exclude files and directories based on the exclusion patterns
    for pattern in exclude_patterns:
        folder_contents = {item for item in folder_contents if pattern not in item}
    to_unzip = zip_contents - folder_contents

    return to_unzip


def extract_all_zip_files_in_tree_recursively(
    path,
    source_root=params["source_folder"],
    export_root=params["export_folder"],
    current_stack=(params["source_folder"],),
    remove_old=False,
):
    filelist = os.listdir(path)
    for file_or_folder in filelist:
        if file_or_folder.endswith(".zip"):
            file = file_or_folder
            zip_file = os.path.join(path, file)
            if current_stack[-1].startswith("Env"):
                new_folder = os.path.join(*current_stack)
            else:
                new_folder = os.path.join(*current_stack, file)[:-4]
            new_folder = new_folder.replace(source_root, export_root)
            to_unzip = None
            if os.path.exists(new_folder):
                if remove_old:
                    shutil.rmtree(new_folder)
                else:
                    to_unzip = compare_directory_structure(
                        zip_file, new_folder, exclude_patterns=("__MACOSX", ".DS_Store")
                    )
                    if len(to_unzip) == 0:
                        logging.info(f"Skipping {zip_file} as it has been extracted")
                        continue
                    else:
                        logging.info(f"Found {len(to_unzip)} new objects to unzip")
                        logging.info(f"Extracting {to_unzip} to {new_folder}")
            os.makedirs(new_folder, exist_ok=True)
            try:
                logging.info(f"Extracting {zip_file} to {new_folder}")
                with zipfile.ZipFile(zip_file, "r") as zip_ref:
                    zip_ref.extractall(new_folder, members=to_unzip)
            except zipfile.BadZipFile as e:
                logging.error(f"Failed to extract {file}: Error: {e}")
        else:
            extract_all_zip_files_in_tree_recursively(
                os.path.join(path, file_or_folder),
                source_root,
                export_root,
                current_stack + (file_or_folder,),
                remove_old,
            )


def prune_directory_tree(root_folder=params["export_folder"]):
    """
    Go over the directory and ensure each folder has the same pattern of subfolders:
    - Task/Home/Env
    And symlink the r3d files to zip files in that directory.
    """
    target_files = []
    for root, dirs, files in os.walk(root_folder):
        if root == root_folder:
            continue
        relative_path = os.path.relpath(root, root_folder)
        if relative_path.count("/") != 2:
            continue
        task, home, env = relative_path.split("/")
        if not env.startswith("Env"):
            logging.warning(f"Found {relative_path} which does not start with Env")
            continue
        for file in glob.iglob(
            os.path.join(root_folder, relative_path, "**/*.r3d"),
            recursive=True,
        ):
            source_file = file
            filename = os.path.basename(file)
            target_file = os.path.join(
                root_folder, relative_path, filename[:-4] + ".zip"
            )
            target_files.append(target_file)
            if not os.path.exists(target_file):
                os.symlink(source_file, target_file, target_is_directory=False)
    return target_files


if __name__ == "__main__":
    os.makedirs(export_folder, exist_ok=True)
    extract_all_zip_files_in_tree_recursively(path=params["source_folder"])
    directory_tree = prune_directory_tree()
    logging.info(f"Found {len(directory_tree)} r3d files")
    # Now save the list of files to a text file
    with open(os.path.join(export_folder, "r3d_files.txt"), "w") as f:
        json.dump(directory_tree, f, indent=4)
    logging.info("Process completed!")
