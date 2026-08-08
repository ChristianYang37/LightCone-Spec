# SGLang patch workflow

[中文](../zh-CN/sglang-patches.md) · [Home](../../README.md)

SGLang is an external Apache-2.0 dependency. LightCone-Spec pins upstream commit
`3312645a307453893a00778592f105581e3d1c3d` and publishes only a mail-formatted
patch series under `patches/sglang/`.

Apply the series only to an exact, clean checkout:

```bash
git clone https://github.com/sgl-project/sglang.git /tmp/sglang
git -C /tmp/sglang checkout --detach 3312645a307453893a00778592f105581e3d1c3d
patches/sglang/apply.sh /tmp/sglang
```

`apply.sh` verifies every patch SHA-256 and the final Git tree. `verify.sh`
repeats application in a temporary clone, compiles the changed Python surface,
runs focused tests, and checks a clean reverse checkout. New SGLang changes must
be authored patch-first; do not commit a source checkout or submodule here.
