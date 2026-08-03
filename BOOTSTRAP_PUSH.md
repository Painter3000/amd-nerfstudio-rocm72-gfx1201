# Bootstrap push

This archive is a Git-ready public documentation tree.

For an empty remote repository:

```bash
git clone https://github.com/Painter3000/amd-nerfstudio-rocm72-gfx1201.git
cd amd-nerfstudio-rocm72-gfx1201
cp -a /path/to/amd-nerfstudio-rocm72-gfx1201_bootstrap_v1/. .
git add .
git commit -m "Initialize public RDNA4 Nerfstudio qualification repository"
git push origin main
```

Before pushing, review the tree and confirm that the intended Git identity is configured.
