## Streamline - SVC

A custom web UI for ACE-Step 1.5 and Applio RVC. Singing Voice Conversion by combining both projects; ACE-Step for timbre and style adaption and RVC for the final higher quality output.

Built on a FastAPI backend and an Electron-based web UI.

<img width="2560" height="1307" alt="preview" src="https://github.com/user-attachments/assets/95ca94ad-18ce-4196-9922-3c80d3b0bc73" />


## Requirements

- A working [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) installation


## Installation

```
cd <your ACE-Step 1.5 install folder>
git clone --recurse-submodules https://github.com/dopf-26/streamline_svc.git
```

Run start_vocals.bat to create a virtual environment and install the requirements using UV automatically.


## Usage

Make sure that you started ACE-Step 1.5 and Applio RVC at least once before and have downloaded your favorite models.

Run `start_vocals.bat` to start both the ACE-Step 1.5 API server and the Streamline SVC frontend.

LoRA's for Ace-Step will be loaded from a subfolder "lora" inside your ace-step folder (e.g. ace-step-1.5/lora/).
Models and Index files for RVC are loaded from streamline_svc/applio/logs.

Remix Strength and Cover Strength are the most important factors on how close the input audio is followed. The defaults (0.05 - 0.27) follow closely but give a bit of creative freedom. I'd suggest to keep Remix Strength at 0.05 and increase cover strength to follow the input more closely, or lower it to be more creative.


## Features

- Sleek modern three stage design
- Automatic input audio transcription using Whisper 3
- Preprocessing Pitch shift, Low Cut and Noise Gate


## Credit and Thanks

- Side-Step theme based on [koda-dernet/Side-Step](https://github.com/koda-dernet/Side-Step)
- [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5)
- [Applio RVC](https://github.com/IAHispano/Applio)
