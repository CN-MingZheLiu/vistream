# VIStream

Core source code and research website for VIStream, including the paper, research film, experimental results, and interactive trajectory demonstrations.

Website: [https://cn-mingzheliu.github.io/vistream/](https://cn-mingzheliu.github.io/vistream/)

Source repository: [CN-MingZheLiu/vistream](https://github.com/CN-MingZheLiu/vistream). The repository's Actions runs show the current deployment status.

## Core code release

[Download the core code: core-code-20261014](https://github.com/CN-MingZheLiu/vistream/releases/tag/core-code-20261014)

[Browse the core source in this repository](https://github.com/CN-MingZheLiu/vistream/tree/main/core), including the method, configurations, evaluation tools, and English setup documentation. The core/ directory contains the source from the pinned release; the repository root contains the research website.

This is a **source-only release** containing the core method code, configurations, and evaluation tools. It is separate from the website source. Datasets, model checkpoints, the Python/CUDA environment, and compiled runtime dependencies must be obtained or installed separately. The archive does not include a ready-to-run offline environment or precomputed prediction trajectories.

Download the source archive and read its included documentation before configuring the required external assets. See [the English core code guide](assets/reproduction-guide.txt) for the release scope, setup requirements, and interpretation of the reported results. The source archive alone is not a self-contained inference installation.

An earlier full-runtime validation completed a fresh run of all 11 KITTI sequences and 132 stages, with mean Sim(3) ATE of **15.25 m with loop closure off** and **9.10 m with loop closure on**. It did **not** exactly reproduce the earlier repaired **15.08 / 10.41 m** result. That later offline repair configuration is distinct from the original paper cohort. These measurements document the earlier configured run; they do not establish that extracting the source-only release reproduces those numbers without the required external assets and setup.

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
- The browser demonstrations replay recorded trajectories. The core code release requires separately provisioned datasets, checkpoints, and runtime dependencies before fresh inference; the web demo itself does not run model inference.
- QA/STATIC_VALIDATION.json records static-link checks; QA/ASSET_PRESERVATION.json records SHA256 comparisons with the original website at migration time; QA/BUILD_PROJECT_PATH_TEST.json records project-path build checks. These are migration snapshots, not verification reports for later documentation edits.

The site requires no npm packages, third-party CDN, personal access token, or server-side application. Its GitHub workflow uses platform-provided permissions; personal credentials must not be committed to the source.
