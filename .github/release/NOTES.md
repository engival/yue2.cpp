Bug-fix release of **yue2.cpp**: YuE2-3B song generation (lyrics + style prompt → full-length 48 kHz stereo FLAC) on ggml/Vulkan. No PyTorch, any GPU with a Vulkan driver.

## Fixed

- **Windows: `yue2 song` failed at the NAR stage** with `truncated tensor data for 'blk.25.attn_k.weight'` (#4). The NAR reads the AR GGUF with its own loader, which seeked with a 32-bit `long` on Windows, so any tensor past 2 GiB in `yue2-ar-q8_0.gguf` could not be read. Seeks are 64-bit now (also in the VAE and `--nar-lora` loaders). Linux was not affected.

## New since v0.1.0

- **Request text from files**: `"lyrics_file"`, `"abc_file"` and `"abc_template_file"` read plain UTF-8 text files in place of `"lyrics"` / `"abc"` / `"abc_template"`, so no `\n` escapes. Relative paths resolve against the request file. CRLF and a UTF-8 BOM are dropped.
- **`examples/make-song`** (`.sh` for Linux, `.ps1` / `.bat` for Windows), bundled in the archives: edit the name, style and lyrics at the top, run it, get `examples/NAME/NAME.flac`.

## Downloads

| File | For |
|---|---|
| `yue2-@VERSION@-linux-x86_64.tar.gz` | Linux x86_64, glibc 2.35+ (Ubuntu 22.04 or newer and equivalents). One static binary; needs only `libvulkan.so.1`. |
| `yue2-@VERSION@-windows-x64.zip` | Windows 10/11 x64. `yue2.exe` + FLAC/ogg DLLs; needs a Vulkan driver and the MSVC x64 redistributable. |
| `SHA256SUMS.txt` | Checksums for both. |

Each archive has a `README-QUICKSTART.txt`. Converted GGUFs from v0.1.0 are still valid; there is no need to convert again.

## Quick start

```
hf download m-a-p/YuE2-3B
hf download m-a-p/YuE2-Vae
yue2 song --request song.json --out song.flac --gpu 0
```

The first `yue2 song` converts the Hugging Face weights to GGUF next to the binary (one time, CPU, under a minute). See the [README](https://github.com/engival/yue2.cpp#readme) for the request format, scores, style changes and every option.

## Known limits

- The Windows build is CI-built; it is tested on real hardware only through user reports. Please open an issue, good or bad.
- Tested on Linux with AMD (RADV) and Intel (ANV) GPUs. On NVIDIA (Windows), the AR stage has been reported working (#4); a full render is not confirmed yet.
- The release builds write FLAC only; Opus output needs a build from source with libopusenc.

## Licences

yue2.cpp is MIT. The YuE2 model weights (m-a-p/YuE2-3B, m-a-p/YuE2-Vae) and the EAGLE-3 draft head are **CC BY-NC 4.0: non-commercial use only**. Third-party code is listed in `NOTICE.md` and `licenses/` inside each archive.
