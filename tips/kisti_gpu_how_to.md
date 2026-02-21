# Contents
1. [How to log on?](#How-to-Log-On-to-the-KISTI-Cluster)
2. [Set up the environment]()

## How to Log On to the KISTI Cluster?

1. Run the following command from the Linux prompt:
```bash
$ssh -X x3397a01@neuron.ksc.re.kr
```

 - [Tip] Instead of doing this, we may create the following ssh config file:
```
# 1. KSC Neuron server (including -X option)
Host neuron
    HostName neuron.ksc.re.kr
    User x3397a01
    Port 22
    ForwardX11 yes
```
Please put this file under ~/.ssh/config. After doing this, you only need to run
the following command:
```bash
$ssh neuron
```
 - [Tip]
Run the following command to log on without typing the ssh password every time:
```
ssh-keygen -t ed25519
```

```
ssh-copy-id -i ~/.ssh/id_ed25519.pub neuron
```

2. Type the OTP password and password.
 - Use AnyOTP to get your current OTP password.

 - Tip: If you 


## Set Up the Environment.
1. Move to the work directory by typing the following command:
```bash
cds
```

# More tips
 - Operating System: Ricky Linux

# Useful links:
 - Notion article writteen by Yanghun Ham:
     : https://debonair-editor-3de.notion.site/Kisti-gpu-2e3100011f688073b102f65dff13f0e1?source=copy_link

# Cheat Sheet
