# KISTI Cluster Quick Start Guide

## Table of Contents
1. [How to Log On](#1-how-to-log-on)
2. [Environment Setup](#2-environment-setup)
3. [Code Development and Debug](#3-code-development-and-debug)
4. [Launch Job]
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
    1. **OTP Password**: Use the **AnyOTP** app to get your current code.
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
- Directly log on to the debug servers (gdebug01 or gdebug02)
```bash
 ssh gdebug01 or gdebug02
```
  - As of Feb. 2026, debug nodes are equipped with V100 with 16 GB VRAM.

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

`transformers[torch] ` will 
