 KISTI Cluster Quick Start Guide

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
# 💡 Tip: Simplify with SSH ConfigInstead of typing the full address every time, 
you can create an SSH configuration file. 
Add the following block to your local `~/.ssh/config` file:Plaintext

```
Host neuron
    HostName neuron.ksc.re.kr
    User x3397a01
    Port 22
    ForwardX11 yes
```
After saving the file, you only need to run:Bashssh neuron

---
# 💡  Tip: Passwordless Login (SSH Keys)
To log on without typing your password every time, use an Ed25519 key pair:
Generate the key:
```bash
ssh-keygen -t ed25519
```
Copy the key to the server:
```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub neuron
```
---

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

### 3. Useful Tips & Links
  -  Operating System: Rocky Linux
  -  Notion Guide: Written by Yanghun Ham
    : https://debonair-editor-3de.notion.site/Kisti-gpu-2e3100011f688073b102f65dff13f0e1?source=copy_link
📝 Cheat Sheet

Command Description
ssh neuronConnect to the clustercds
Move to work directory
exit  Logout from the session
