# Getting started with Dobb·E code

In this section, we will help you get started with running all the software parts of Dobb·E. There are three major parts in the software component of Dobb-E, which are:

1. Processing data collected with the Stick,
2. Training a model on the collected data,
3. Deploying the model.

Optionally, you can [pre-train your own model](optional-training-your-own-home-pretrained-representations.md) similar to how we trained the Home Pretrained Representations (HPR).

We will frequently refer to the "robot", which for this part would be the Intel NUC installed in the Hello Robot Stretch, and the "machine", which should be a beefier machine with GPU(s) where you can preprocess your data and train new models.

## Getting Started

*   Clone the repository:

    ```bash
    git clone https://github.com/notmahi/dobb-e.git
    cd imitation-in-homes
    ```
*   Set up the project environment:

    ```bash
    mamba env create -f conda_env.yaml
    ```
*   Activate the environment:

    ```bash
    mamba activate home_robot
    ```
* Logging:
  *   To enable logging, log in with a Weights and Biases (`wandb`) account:

      ```bash
      wandb login
      ```
  *   Alternatively, disable logging altogether:

      ```bash
      export WANDB_MODE=disabled
      ```

## Setting up the Datasets

Download our provided datasets or get started with your own data.

### **Download Our Datasets**

You can simply download our pre-provided datasets using any of the following commands:

```bash
# HoNY RGB + actions dataset, 814 MB
wget https://dl.dobb-e.com/datasets/homes_of_new_york.zip
unzip homes_of_new_york.zip

# HoNY RGB-D + actions dataset, 77 GB
pip install gdown
python -c "import gdown; gdown.download_folder('https://drive.google.com/drive/folders/1o8c6b6hSKfId8EzemVGf8c7DQoZ2IHAO?usp=sharing', quiet=True)"
zip -FF HomesOfNewYorkWithDepth.zip --out HoNYDepth.zip
unzip -FF HoNYDepth.zip

# Sample finetuning dataset
wget https://dl.dobb-e.com/datasets/finetune_directory.zip
unzip finetune_directory.zip
```

### **Bring Your Own Data**

Follow documentation in [stick-data-collection](../stick-data-collection) to extract data.

*   Ensure the following data directory structure:

    <pre><code><strong>dataset/
    </strong>|--- task1/
    |------ home1/
    |--------- env1/
    |--------- env1_val/
    |--------- env2/
    |--------- env2_val/
    |--------- env.../
    |------ home2/
    |--------- env1/
    |--------- env1_val/
    |--------- env.../
    |------ home.../
    |--- task2/
    |------ home1/
    |--------- env1/
    |--------- env1_val/
    |--------- env.../
    |------ home2/
    |--------- env1/
    |--------- env1_val/
    |--------- env.../
    |------ home.../
    |--- task.../
    |--- r3d_files.txt
    </code></pre>

### Create the env\_vars file

Create the file `configs/env_vars/env_vars.yaml` based on `configs/env_vars/env_vars.yaml.sample`.
`home_ssl_data_root` and `home_ssl_data_original_root`:  These are optional, only relevant if you are planning to reproduce the HPR encoder.

<details>
  <summary>
    <h2>[Optional] Training Your Own Home Pretrained Representations</h2>
  </summary>
  
This step assumes you have already downloaded the HoNY RGB + actions dataset to somewhere on your machine, and have updated your `configs/env_vars/env_vars.yaml` accordingly.

*   **Single GPU:** To reproduce our HPR encoder, run in terminal:

    ```bash
    python train.py --config-name=train_moco
    ```
*   Multi-GPU: We use huggingface 🤗 accelerate to run multi-GPU training. Run the following commands

    ```bash
    accelerate config # Interactively walk you through setting up multi-GPU training
    accelerate launch train.py --config-name=train_moco
    ```

</details>

## Training Policies

The following assumes that the current working directory is this repository’s root folder.

### Training a Behavior Cloning Policy

1. Modify `include_task` and `include_env` in `finetune.yaml` depending on the task and env you intend to finetune.
2. \[Optional, only if you're using torch encoder] Set `enc_weight_pth` (path to pretrained encoder weights) in `image_bc_depth.yaml`.
3.  Run in terminal:

    ```bash
    python train.py --config-name=finetune
    ```
## Deploying a Policy on the Robot

1. Follow “Getting Started” in the [`robot-server`](../robot-server) documentation.
   * Skip the dataset related parts if you're not running VINN
  
2.  Install ROS1 within your Conda environment:

    ```bash
    # Only if you are not using mamba already
    conda install mamba -c conda-forge
    # this adds the conda-forge channel to the new created environment configuration 
    conda config --env --add channels conda-forge
    # and the robostack channel
    conda config --env --add channels robostack-staging
    # remove the defaults channel just in case, this might return an error if it is not in the list which is ok
    conda config --env --remove channels defaults

    mamba install ros-noetic-desktop
    ```

    Reference: [https://robostack.github.io/GettingStarted.html](https://robostack.github.io/GettingStarted.html)
3. Follow documentation in [robot-server](../robot-server) for running `roscore` and `start_server.py` on the robot.
4. Ensure that both of the previous commands are both running in the background in their own separate windows. We like using `tmux` for running multiple commands and keeping track of them at the same time.

### Behavior Cloning

1. Transfer over the weights of a trained BC policy to the robot.
   1.  Take the last checkpoint (saved after 50th epoch):

       ```bash
       rsync -av --include='*/' --include='checkpoint.pt' --exclude='*' checkpoints/2023-11-22 hello-robot@{ip-address}:/home/hello-robot/code/imitation-in-homes/checkpoints
       ```
2. In `configs/run.yaml` set `model_weight_pth` to the path containing the trained BC policy weights.
3.  Run in terminal:

    ```bash
    python run.py --config-name=run
    ```

### VINN

1. Transfer over the encoder weights and `finetune_task_data` onto the robot.
   1. We recommend doing so using `rsync`
   2.  To speed up the transfer of data and save space on the robot, only transfer the necessary files:

       ```bash
       rsync -avm --include='*/' --include='*.json' --include='*.bin' --include='*.txt' --include='*.mp4' --exclude='*' /home/shared/data/finetune_task_data hello-robot@{ip-address}:/home/hello-robot/data
       ```
2. In `configs/run_vinn.yaml` set `checkpoint_path` to encoder weights.
3. In `configs/dataset/vinn_deploy_dataset.yaml`, set `include_tasks` and `include_envs` to be a specific task (i.e. Drawer\_Closing) and environment (i.e. Env2) from the `finetune_task_data` folder.
4.  Run in terminal:

    ```bash
    python run.py --config-name=run_vinn
    ```
