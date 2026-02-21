# KISTI Cluster Quick Start Guide

## Table of Contents
1. [How to Log On](#1-how-to-log-on)
2. [Environment Setup](#2-environment-setup)
3. [Useful Tips & Links](#3-useful-tips--links)

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
Host neuron
    HostName neuron.ksc.re.kr
    User x3397a01
    Port 22
    ForwardX11 yes
```
After saving the file, you only need to run:
```bash
ssh neuron
```

### 💡  Tip: Passwordless Login (SSH Keys)

Passwordless login via public keys is disabled on the KISTI cluster. 
Therefore, you cannot bypass the password prompt using RSA or Ed25519 keys.

## 2. Environment Setup

### Authentication
When prompted, enter your credentials in this order:
1. **OTP Password**: Use the **AnyOTP** app to get your current code.
2. **Account Password**: Your standard KISTI account password.


### Navigating to Work Directory
Move to your work directory instantly by typing:

```bash
cds
```
---
## 3. Useful Tips & Links
  -  Operating System: Rocky Linux
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
