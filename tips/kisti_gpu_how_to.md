# KISTI Cluster Quick Start Guide

## Table of Contents
1. [How to Log On](#1-how-to-log-on)
2. [Environment Setup](#2-environment-setup)
3. [Code Development and Debug](#3-code-development-and-debug)
4. [Launch Job](#4-job-launch)
5. [Useful Tips & Links](#5-useful-tips--links)
6. [Example of Making a Conda Environment]

---

## 1. How to Log On

### Standard Method
Run the following command from your Linux prompt:
```bash
ssh -X x3397a01@neuron.ksc.re.kr
```

---

### 💡 Tip: Simplify with SSH Config

Instead of typing the full address every time, 
you can create an SSH configuration file. 
Add the following block to your local `~/.ssh/config` file:Plaintext

```
# 1. KSC Neuron server (including -X option)
Host neuron
    HostName neuron.ksc.re.kr
    User x3397a01
    Port 22
    ForwardX11 yes

# 2. KSC Neuron Datamover server (including -X option)
Host neuron-dm
    HostName neuron-dm.ksc.re.kr
    User x3397a01
    Port 22
    ForwardX11 yes
```
After saving the file, you only need to run:
```bash
ssh neuron
```

### 💡  Tip: Datamover server
 - The CPU limit in the login server is 2 hours. Therefore, when you copy data, please use the datamover server.

   
### 💡  Tip: Passwordless Login (SSH Keys) Is **Disabled**

Passwordless login via public keys is **disabled** on the KISTI cluster. 
Therefore, you **cannot** bypass the password prompt using RSA or Ed25519 keys.

## 2. Environment Setup

### Authentication
- When prompted, enter your credentials in this order:
    1. **OTP Password**: Use the **AnyOTP** app to get your current one-time password.
    2. **Account Password**: Your standard KISTI account password.


### Navigating to Work Directory
- Move to your work directory instantly by typing:

```bash
cds
```

### Moving Data to the Cluster
- Copy the training or testing data using the **Datamover** node. The following is an example:
  ```bash
  scp -r libri_light/ neuron-dm:/scratch/x3397a01/chanwcom/database
  ```
---
### Creating and activating Conda Environment
 - 💡 You do NOT need to install conda binary since it has been already set up.
 - Just create an appropriate conda environment:
   ```bash
   conda create --name py3_10_hf python=3.10
   conda activate py3_10_hf
   ```

## 3. Code Development and Debug



### Debug
- Directly log on to the debug nodes (gdebug01 or gdebug02)
```bash
 ssh gdebug01 or gdebug02
```
  - As of Feb. 2026, debug nodes are equipped with V100 with 16 GB VRAM.
  - In the debug node, a very limited amount of VARM will be assigned to you, so please reduce the batch size.
    (A rule of thumb is that when you change the batch size, you also need to change the learning rate in proportion).
---
## 4. Job Launch
 - If you think your code runs smoothly when tested in the debug node (that is mentioned in the previosu section),
   submit the job using the following command:
``` bash
   sbatch job.sh
```
 - The following shows an example of `job.sh`
```
#!/bin/bash


#SBATCH -J shc_48_batch_org_ctc
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o /scratch/x3397a01/chanwcom/experiments/log/%x_%j.out
#SBATCH -e /scratch/x3397a01/chanwcom/experiments/log/%x_%j.err
#SBATCH --time=0-04:00:00 #4 hours
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --comment=python

HOME=chanwcom
# Set up the environment
module purge
module load cuda/12.9.1  # Use PyTorch cu129 build
source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate py3_10_hf

# Run your code (This is a single GPU job case)
cd /scratch/x3397a01/$HOME/local_repository/cognitive_workflow_kit/run
#srun python wav2vec_finetuning_shc.py --vocab_size=32
python wav2vec_finetuning_shc.py --vocab_size=32
```

## 5. Useful Tips & Links
  -  Operating System: Rocky Linux
  -  Official Neuron Guide
    : https://docs-ksc.gitbook.io/neuron-user-guide/undefined/user-environment
  -  Notion Guide: Written by Yanghun Ham
    : https://debonair-editor-3de.notion.site/Kisti-gpu-2e3100011f688073b102f65dff13f0e1?source=copy_link

---
## 4. 📝 Cheat Sheet

|    Command |       Description            | Comments  |
|------------|------------------------------|-----------|
| ssh neuron | Connect to the cluster       |           |
| cds        | Move to work directory       |           |
| quotainfo  | Show Filesystem Quota Status |           |
| isam       | Check the remaining SRU time   |     https://www.youtube.com/watch?v=1k6Gpke54Uk      |
| overview_nodes  | Show the status of the cluser |           |

| Category | Command | Description | Example / Usage |
| :--- | :--- | :--- | :--- |
| **Submission** | `sbatch [file]` | Submit a batch script to the queue | `sbatch train.sh` |
| **Submission** | `srun --pty bash` | Interactive login to a compute node | `srun -p amd_a100nv_8 --gres=gpu:1 --pty bash` |
| **Monitoring** | `squeue` | View all active jobs in the system | `squeue` |
| **Monitoring** | `squeue -u $USER` | View only **your** jobs | `squeue -u $USER` |
| **Monitoring** | `sinfo` | View partition and node status | `sinfo -p amd_a100nv_8` |
| **Management** | `scancel [id]` | Cancel a specific job | `scancel 12345` |
| **Management** | `scancel -u $USER` | Cancel **all** of your jobs | `scancel -u $USER` |
| **Details** | `scontrol show job` | Show detailed job configuration | `scontrol show job 12345` |
| **History** | `sacct` | Display accounting data for past jobs | `sacct -j 12345` |
| **Quota** | `sshare` | Check your remaining budget/priority | `sshare -U $USER` |
| **Priority** | `sprio` | View factors affecting job priority | `sprio -j 12345` |


## 6. Example of Making a Conda Environment
```
conda create --name py3_10_hf python=3.10
conda activate py3_10_hf
```
```
module load cuda/12.9.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip3 install transformers[torch] datasets evaluate torchcodec
pip3 install sentencepiece
pip3 install webdataset
```

`transformers[torch] ` will install extra tools for torch.

## 7. Installing extra programs

- The following is not needed if you do NOT need to run neovim, xclip or zoxide.

### nvim
```
curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-linux-x86_64.appimage
```

### xclip
```
TMPDIR=/scratch/x3397a01/tmp dnf download xclip
```
```
rpm2cpio xclip-0.13-17.git11cba61.el9.x86_64.rpm | cpio -idmv
```
```
cp usr/bin/xclip ~/.local/bin
```

### zoxide
```
cd ~/tmp
# Downloading the version 0.9.4 (The size is between 1MB~2MB)
curl -L -o zoxide.tar.gz https://github.com/ajeetdsouza/zoxide/releases/download/v0.9.4/zoxide-0.9.4-x86_64-unknown-linux-musl.tar.gz

# 1. Decompressing
tar -xzvf zoxide.tar.gz

# 2. Moving the binary
mv zoxide ~/.local/bin/

# 3. Checking the file size
ls -l ~/.local/bin/zoxide

# 4. Adding information to ./bashrc
echo 'eval "$(zoxide init bash)"' >> ~/.bashrc
```

```
source ~/.bashrc
```

