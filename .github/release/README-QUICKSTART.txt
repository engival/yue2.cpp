yue2.cpp @VERSION@ - quick start
================================

yue2 turns lyrics + a style prompt into a full song (48 kHz stereo FLAC) with
YuE2-3B, on any GPU with a Vulkan driver. No Python is needed.

Full manual: https://github.com/engival/yue2.cpp#readme
Bug reports: https://github.com/engival/yue2.cpp/issues


1. What you need
----------------
- A GPU with an up-to-date Vulkan driver (the binary needs only the system
  Vulkan loader: libvulkan.so.1 on Linux, vulkan-1.dll on Windows, which every
  Vulkan driver installs). `vulkaninfo --summary` should list your GPU.
- A 64-bit x86 CPU with AVX2 (2013 or newer).
- About 6 GB free next to the yue2 binary for the converted model files,
  plus the downloaded weights (~7.5 GB in the Hugging Face cache).
- Windows only: the Microsoft Visual C++ Redistributable (x64), which most
  machines already have.


2. Get the weights (CC BY-NC 4.0, non-commercial - see NOTICE.md)
-----------------------------------------------------------------
With the Hugging Face CLI (pip install -U huggingface_hub):

    hf download m-a-p/YuE2-3B
    hf download m-a-p/YuE2-Vae

They land in the Hugging Face cache (~/.cache/huggingface/hub, or
%USERPROFILE%\.cache\huggingface\hub on Windows, or wherever HF_HOME /
HF_HUB_CACHE point), where yue2 finds them on its own.


3. Convert (optional)
---------------------
    yue2 convert

writes yue2-ar-q8_0.gguf, yue2-nar-f16.gguf and yue2-vae-f32.gguf next to the
binary (~35 s, CPU only). You can skip this: the first `yue2 song` converts
whatever is missing before it renders.


4. Write a request, e.g. song.json
----------------------------------
    {
      "style": "slow dream pop, reverb guitar, brushed drums, breathy female vocal",
      "lyrics": "[Verse]\nThe kettle sings a flat blue note\n\n[Chorus]\nStay a while, the rain is warm\n",
      "seed": 1
    }


5. Render
---------
    yue2 song --request song.json --out song.flac --gpu 0

`--gpu N` picks the Vulkan device. The very first run compiles the GPU
pipelines for your driver and can take 15 minutes or more; later runs start fast.
`yue2 song --help` lists every option.


Notes
-----
- The Linux build is made on Ubuntu 22.04 (glibc 2.35 or newer).
- The Windows build is produced by CI and is NOT tested by the maintainer on
  real hardware. Reports, good or bad, are very welcome.
- Licences: yue2.cpp is MIT (LICENSE); third-party code and the model weights'
  terms are listed in NOTICE.md and licenses/.
