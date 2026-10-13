#!/usr/bin/env python3
"""Assemble only public site assets and inject the real Pages deployment URL."""
import argparse
import html
import shutil
from pathlib import Path
from urllib.parse import urlsplit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    base = args.base_url.rstrip('/')
    url = urlsplit(base)
    if url.scheme != 'https' or not url.netloc or url.query or url.fragment or url.username or url.password:
        parser.error('--base-url must be an HTTPS origin plus optional project path')
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output == root or output in root.parents:
        parser.error('output must not replace source or a parent directory')
    if output.exists():
        parser.error('output exists; choose a fresh output directory')
    output.mkdir(parents=True)
    for name in ('index.html', 'style.css', 'app.js', '.nojekyll'):
        shutil.copy2(root / name, output / name)
    shutil.copytree(root / 'assets', output / 'assets')
    page = output / 'index.html'
    markup = page.read_text(encoding='utf-8')
    marker = '<!-- github-pages:deployment-metadata -->'
    if markup.count(marker) != 1:
        raise SystemExit('Expected exactly one deployment metadata marker')
    canonical = html.escape(base + '/', quote=True)
    poster = html.escape(base + '/assets/poster.jpg', quote=True)
    markup = markup.replace(marker, f'<link rel="canonical" href="{canonical}">\n  <meta property="og:url" content="{canonical}">')
    markup = markup.replace('property="og:image" content="assets/poster.jpg"', f'property="og:image" content="{poster}"')
    page.write_text(markup, encoding='utf-8')
    print(f'Built {output}; canonical: {base}/')


if __name__ == '__main__':
    main()
