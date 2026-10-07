# Releasing

Releases are built by GitHub Actions from a tag; nothing is uploaded by hand.

## Cutting a release

1. Make sure `master` is green in the **CI** workflow (`.github/workflows/build.yml`)
   and the `llama.cpp` submodule points at a commit that is public on the fork.
2. Tag and push the tag:

   ```bash
   git tag -a v0.1.0 -m "yue2.cpp v0.1.0"
   git push origin v0.1.0
   ```

   A tag with a `-` in it (`v0.2.0-rc1`) becomes a pre-release.
3. The **Release** workflow (`.github/workflows/release.yml`) builds
   `yue2-<tag>-linux-x86_64.tar.gz` (Ubuntu 22.04, glibc 2.35+) and
   `yue2-<tag>-windows-x64.zip` (MSVC), runs the smoke tests on both, and creates a
   **draft** release with the two archives and `SHA256SUMS.txt`, plus
   auto-generated notes.
4. Open the draft under *Releases*, download and try the Linux archive on a real
   GPU (`yue2 song` end to end), edit the notes, then **Publish**.
5. A failed run can be re-run from the Actions tab; it replaces the draft's files.
   To redo a tag from scratch, delete the draft and the tag
   (`git push origin :refs/tags/v0.1.0`) and push it again.

Each archive holds the `yue2` binary, `LICENSE`, `NOTICE.md`, `licenses/`
(libFLAC, and on Windows libogg) and `README-QUICKSTART.txt` (from
`.github/release/`, `@VERSION@` replaced by the tag). The Windows zip also carries
the vcpkg DLLs `yue2.exe` imports (FLAC, ogg); `vulkan-1.dll` and the MSVC
runtime come from the user's system.

## Build configuration

| | Linux | Windows |
|---|---|---|
| runner | ubuntu-22.04 | windows-latest |
| Vulkan SDK | LunarG jammy apt repo | LunarG installer (pinned version) |
| ggml / llama | static (`BUILD_SHARED_LIBS=OFF`), `GGML_NATIVE=OFF` (AVX2 baseline), `GGML_OPENMP=OFF` | same |
| libFLAC | 1.5.0 built from the pinned upstream tarball, static | vcpkg `x64-windows`, DLL |
| C++ runtime | `-static-libstdc++ -static-libgcc` | dynamic MSVC CRT |
| Opus | off | off |
| EAGLE-3 | on | on |

The Linux release job fails if `yue2` links anything beyond `libvulkan.so.1` and glibc.

## What CI checks — and what it doesn't

Runners have no GPU and no model weights. On every push to `master` and every
pull request, `build.yml` builds:

- Linux, Ubuntu 22.04, release flags, `YUE2_EAGLE3=ON` and `OFF`;
- Linux, Ubuntu 24.04, distro packages, shared ggml, `YUE2_OPUS=ON`;
- Windows, MSVC x64, release flags.

and then `.github/scripts/smoke-test.sh`:

- `yue2` with no arguments prints its usage and exits 1; `yue2 song --help` exits 0;
- `yue2 noise` reproduces `tests/golden/noise_meta.json` byte for byte (a warning
  only on Windows, where the CRT's libm may differ);
- the model-free test targets: `yue2-numpy-exp`, `yue2-bars`, `yue2-guidance`,
  `yue2-handover`, `yue2-draft-accept`, `yue2-sampler-diff` (20k cases).

**Not checked:** anything that touches a GPU or the weights — `yue2 convert`
byte-identity, AR/NAR/VAE goldens, `tests/regress.sh`, audio quality, speed, and
whether the Windows binary renders a song at all. Those are the maintainer's
manual checks before publishing a draft.
