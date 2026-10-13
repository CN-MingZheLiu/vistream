#!/usr/bin/env python3
"""Validate local resources without fetching or changing experimental data."""
import argparse
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        for key in ('href', 'src', 'poster'):
            if key in attrs:
                self.refs.append((tag, key, attrs[key]))
        if tag == 'meta' and attrs.get('property') == 'og:image':
            self.refs.append((tag, 'content', attrs['content']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    root = args.site.resolve()
    parsed = References()
    parsed.feed((root / 'index.html').read_text())
    refs = list(parsed.refs)
    for name in ('app.js', 'style.css'):
        text = (root / name).read_text()
        refs.extend((name, 'resource', item) for item in re.findall(r'''["'](assets/[^"']+)["']''', text))
        refs.extend((name, 'css-url', item.strip('"\'')) for item in re.findall(r'url\(([^)]+)\)', text))
    errors = []
    local = set()
    external = set()
    for source, attr, value in refs:
        if value.startswith('#'):
            if value[1:] and value[1:] not in parsed.ids:
                errors.append(f'Missing anchor: {value}')
            continue
        url = urlsplit(value)
        if url.scheme or url.netloc:
            external.add(value)
            continue
        if value.startswith('/'):
            errors.append(f'Root-absolute URL would fail under a project path: {value}')
            continue
        path = (root / unquote(url.path)).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            errors.append(f'Missing/unsafe local resource ({source} {attr}): {value}')
        local.add(url.path)
    for path in root.rglob('*'):
        if not path.is_file() or path.suffix not in {'.html', '.css', '.js', '.json', '.txt', '.svg', '.vtt'}:
            continue
        text = path.read_text(encoding='utf-8')
        if 'chatgpt.site' in text:
            errors.append(f'Old hosting dependency: {path.relative_to(root)}')
        if re.search(r'-----BEGIN [A-Z ]*PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16}', text):
            errors.append(f'Potential credential: {path.relative_to(root)}')
    sizes = {str(p.relative_to(root)): p.stat().st_size for p in root.rglob('*') if p.is_file()}
    report = {'status': 'PASS' if not errors else 'FAIL', 'site': str(root),
              'local_resources_checked': sorted(local), 'external_urls': sorted(external),
              'total_bytes': sum(sizes.values()), 'largest_files': sorted(sizes.items(), key=lambda x: -x[1])[:6],
              'errors': errors,
              'scope': 'Static resource, anchor, project-path portability and common plaintext credential-pattern checks. No remote deployment or browser runtime is claimed.'}
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.report:
        args.report.write_text(output + '\n', encoding='utf-8')
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
