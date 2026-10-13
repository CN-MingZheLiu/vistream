# VIStream · GitHub Pages

Standalone GitHub Pages source for the VIStream research website, including the research film, paper, experimental results, and interactive trajectory demonstrations.

Website: [https://cn-mingzheliu.github.io/vistream/](https://cn-mingzheliu.github.io/vistream/)

Source repository: [CN-MingZheLiu/vistream](https://github.com/CN-MingZheLiu/vistream). The repository's Actions runs show the current deployment status.

## Reproduction package

[KITTI reproduction release: kitti-reproduction-20261012](https://github.com/CN-MingZheLiu/vistream/releases/tag/kitti-reproduction-20261012)

The complete code, checkpoints, and offline environment are distributed as release assets, separately from this website source. Datasets and precomputed prediction trajectories are excluded. The original archive is VIStream_KITTI_Reproduction_20261012.zip (11,167,525,236 bytes).

The release splits that ZIP into six ordered binary parts, VIStream_KITTI_Reproduction_20261012.zip.part01 through .part06, each at most 1,900,000,000 bytes (under 2 GB). **These parts are not independent ZIP files.** Download all six parts and both checksum files into one directory, verify the parts, concatenate them in order, and verify the restored ZIP:

~~~sh
sha256sum -c SHA256SUMS.parts.txt
cat VIStream_KITTI_Reproduction_20261012.zip.part01 \
    VIStream_KITTI_Reproduction_20261012.zip.part02 \
    VIStream_KITTI_Reproduction_20261012.zip.part03 \
    VIStream_KITTI_Reproduction_20261012.zip.part04 \
    VIStream_KITTI_Reproduction_20261012.zip.part05 \
    VIStream_KITTI_Reproduction_20261012.zip.part06 \
    > VIStream_KITTI_Reproduction_20261012.zip
sha256sum -c SHA256SUMS.zip.txt
unzip VIStream_KITTI_Reproduction_20261012.zip
~~~

On macOS, replace sha256sum -c with shasum -a 256 -c. Inference requires Linux and an NVIDIA GPU; macOS can download, verify, and extract the files. See [the full reproduction guide](assets/reproduction-guide.txt) for dataset layout, environment restoration, inference, and evaluation.

The packaged configuration completed a fresh run of all 11 KITTI sequences and 132 stages, with mean Sim(3) ATE of **15.25 m with loop closure off** and **9.10 m with loop closure on**. It did **not** exactly reproduce the earlier repaired **15.08 / 10.41 m** result. This later offline repair configuration is distinct from the original paper cohort; the archive is not a claim of exact numerical reproduction of every paper result.

## Deployment

1. Commit this directory's contents to the main branch of a public vistream repository. Include the hidden files .github/workflows/pages.yml and .nojekyll.
2. In **Settings → Pages → Build and deployment → Source**, select **GitHub Actions**.
3. Under **Actions**, run **Deploy VIStream to GitHub Pages**. Subsequent pushes to main deploy automatically.
4. Once deployment succeeds, open the official URL shown by the workflow or the repository's Pages settings.

The workflow uses GitHub's official Pages Actions. During deployment, configure-pages supplies the actual base_url, which the build script uses for the canonical URL, Open Graph URL, and absolute preview-image URL. Images, videos, scripts, the paper, and data use relative paths, so the site works under /vistream/ or another project path.

## Local preview and validation

Run from this directory:

~~~sh
python3 scripts/verify_site.py
python3 -m http.server 8000
~~~

Then open http://localhost:8000/. Use an HTTP server for previewing; opening the HTML file directly can prevent the browser from loading JSON data.

The build publishes only index.html, app.js, style.css, .nojekyll, and assets/. The README, workflow, QA reports, and helper scripts are excluded from the deployed artifact.

## Content and evidence

- The website retains the original videos, paper, images, subtitles, interactive trajectory data, result tables, and reproduction instructions.
- Experimental data have not been recalculated or altered during the hosting migration. Changes concern deployment metadata and documentation.
- The browser demonstrations replay recorded trajectories. Fresh model inference is performed by the separate reproduction package, not by the web demo.
- QA/STATIC_VALIDATION.json records static-link checks; QA/ASSET_PRESERVATION.json records SHA256 comparisons with the original website at migration time; QA/BUILD_PROJECT_PATH_TEST.json records project-path build checks. These are migration snapshots, not verification reports for later documentation edits.

The site requires no npm packages, third-party CDN, personal access token, or server-side application. Its GitHub workflow uses platform-provided permissions; personal credentials must not be committed to the source.
