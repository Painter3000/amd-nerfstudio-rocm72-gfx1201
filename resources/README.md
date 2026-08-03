# Resource-cache policy

This directory is reserved for immutable setup inputs and their public manifests.
It is not an artifact or checkpoint directory.

Future setup releases may cache resources such as:

- pinned source archives or Git bundles;
- pinned Python wheels;
- a published quick-validation dataset;
- manifests containing source URL, expected size, SHA-256, version, and license.

Generated P1 checkpoints do **not** belong here. They are produced locally inside the evidence run, used for fresh-process reload verification, and deleted by default after successful verification. Use `--keep-checkpoints` only for explicit debugging or evidence retention.

The planned setup behavior is fail-closed:

- interactive mode asks before downloading a missing resource;
- `--auto` may download only entries pinned by the resource manifest;
- `--offline` forbids network access and lists missing cache entries;
- every download is written to a temporary `.part` file, hash-verified, and atomically renamed.

No download manifest or installer is claimed by Public Toolchain v1.2 yet.
