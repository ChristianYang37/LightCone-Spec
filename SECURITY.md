# Security policy

## Supported versions

LightCone-Spec is pre-release software. Security fixes are applied to the
current `main` branch; no stable support window is promised yet.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository. Do
not open a public issue for credential exposure, arbitrary code execution,
unsafe model loading, path traversal, or remote service vulnerabilities.

Include a minimal reproduction, affected commit, platform, impact, and any
suggested mitigation. Remove access tokens, passwords, private prompts, model
credentials, IP addresses, and provider instance details. Maintainers will
acknowledge a valid report and coordinate disclosure after a fix is available.

## Scope notes

Model files, datasets, SGLang, CUDA libraries, and cloud providers are external
dependencies. Reports are welcome when LightCone-Spec uses them unsafely; issues
solely in those projects should also be reported upstream.
