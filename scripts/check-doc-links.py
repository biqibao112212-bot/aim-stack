#!/usr/bin/env python3
"""Validate reader documentation links and every tracked public URL.

The default check is deterministic and validates repository-local files and
GitHub-style heading anchors.  Pass ``--external`` to also resolve every HTTP(S)
URL across every tracked text file (including evidence registries and source
provenance comments), and to query every HTTPS ``.git`` clone endpoint without
prompting for credentials.  Loopback URLs are deliberately reported but not
requested: they only exist while a reader runs the local debug server.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlsplit
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_EXTENSIONS = {".md", ".markdown", ".mdown", ".mkd"}
DOCUMENT_EXTENSIONS = MARKDOWN_EXTENSIONS | {".html", ".htm"}
LINK_PATTERN = re.compile(
    r"!?\[[^\]]*\]\((?P<markdown>[^)]+)\)"
    r"|(?:href|src)=[\"'](?P<html>[^\"']+)[\"']",
    re.IGNORECASE,
)
# Deliberately restrict literal URLs to RFC 3986 URL characters.  It prevents
# Chinese prose immediately following a Markdown link from becoming part of the
# target while still accepting percent-encoded Unicode URLs.
URL_PATTERN = re.compile(
    r"(?P<url>https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+)",
    re.IGNORECASE,
)
FENCE_PATTERN = re.compile(r"^\s*(```|~~~)")
SCHEME_PATTERN = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)
HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
EXPLICIT_ANCHOR_PATTERN = re.compile(
    r"<(?:a\s+(?:[^>]*?\s)?(?:id|name)|[^>]+\s+id)=[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def document_files() -> list[Path]:
    output = subprocess.check_output(["git", "-C", str(REPO_ROOT), "ls-files", "-z"])
    return sorted(
        REPO_ROOT / path.decode("utf-8", errors="surrogateescape")
        for path in output.split(b"\0")
        if path
        and Path(path.decode("utf-8", errors="surrogateescape")).suffix.lower()
        in DOCUMENT_EXTENSIONS
    )


def external_source_files() -> list[Path]:
    """Find every tracked, non-binary file containing an HTTP(S) URL."""
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "grep", "-I", "-l", "-E", r"https?://"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode not in {0, 1}:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git grep failed: {detail}")
    return sorted(
        REPO_ROOT / path.decode("utf-8", errors="surrogateescape")
        for path in result.stdout.splitlines()
    )


def visible_lines(text: str):
    """Yield lines outside fenced code blocks."""
    fence: str | None = None
    for number, line in enumerate(text.splitlines(), 1):
        match = FENCE_PATTERN.match(line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = marker
            elif marker == fence:
                fence = None
            continue
        if fence is None:
            yield number, line


def link_target(raw: str) -> str:
    value = raw.strip()
    if value.startswith("<") and ">" in value:
        return value[1 : value.index(">")]
    # Markdown permits an optional title after a whitespace-separated target.
    return value.split(maxsplit=1)[0]


def github_slug(heading: str) -> str:
    value = re.sub(r"<[^>]+>", "", heading)
    value = re.sub(r"[\\`*_{}\[\]()#+.!,:;?'\"/<>|~@=$%^&]", "", value)
    value = "".join(
        character
        for character in value
        if not unicodedata.category(character).startswith("P")
    )
    value = re.sub(r"\s+", "-", value.strip().lower())
    return value


def anchors(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    values = {unquote(value) for value in EXPLICIT_ANCHOR_PATTERN.findall(text)}
    if path.suffix.lower() not in MARKDOWN_EXTENSIONS:
        return values
    counts: dict[str, int] = {}
    for _, line in visible_lines(text):
        match = HEADING_PATTERN.match(line)
        if not match:
            continue
        base = github_slug(match.group(1))
        count = counts.get(base, 0)
        values.add(base if count == 0 else f"{base}-{count}")
        counts[base] = count + 1
    return values


def local_target(source: Path, target: str) -> tuple[Path, str] | None:
    target = target.strip()
    if not target or target.startswith("//"):
        return None
    if SCHEME_PATTERN.match(target):
        return None
    parsed = urlsplit(target)
    path_text = unquote(parsed.path)
    if not path_text:
        path = source
    elif path_text.startswith("/"):
        path = REPO_ROOT / path_text.lstrip("/")
    else:
        path = source.parent / path_text
    return path, unquote(parsed.fragment)


def clean_url(value: str) -> str:
    """Drop prose punctuation without corrupting ordinary URL query strings."""
    value = value.rstrip(".,;:!?")
    while value.endswith(")") and value.count(")") > value.count("("):
        value = value[:-1]
    return value


def external_references(files: list[Path]) -> dict[str, list[tuple[Path, int]]]:
    """Return all literal and Markdown/HTML HTTP(S) URLs, including code blocks."""
    values: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    for source in files:
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            candidates = [match.group("url") for match in URL_PATTERN.finditer(line)]
            for match in LINK_PATTERN.finditer(line):
                target = link_target(match.group("markdown") or match.group("html") or "")
                if target.lower().startswith(("http://", "https://")):
                    candidates.append(target)
            for candidate in candidates:
                url = clean_url(candidate)
                if url:
                    values[url].append((source, line_number))
    return values


def is_git_endpoint(url: str) -> bool:
    parsed = urlsplit(url)
    return parsed.scheme == "https" and parsed.path.rstrip("/").endswith(".git")


def verify_git_endpoint(url: str, timeout: float) -> str | None:
    environment = os.environ | {"GIT_TERMINAL_PROMPT": "0"}
    try:
        result = subprocess.run(
            ["git", "-c", "credential.interactive=false", "ls-remote", url, "HEAD"],
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"Git repository probe timed out after {timeout:g}s"
    if result.returncode == 0:
        return None
    detail = result.stderr.strip().splitlines()
    return detail[-1] if detail else f"git ls-remote exited {result.returncode}"


def verify_http_url(url: str, timeout: float) -> str | None:
    headers = {"User-Agent": "aim-stack-documentation-link-check/1.0"}
    request_url = quote(url, safe=":/?&=#%+;,@-._~!$'()*[]")
    for method in ("HEAD", "GET"):
        request = Request(request_url, headers=headers, method=method)
        if method == "GET":
            request.add_header("Range", "bytes=0-0")
        try:
            with urlopen(request, timeout=timeout) as response:
                if 200 <= response.status < 400:
                    return None
                return f"HTTP {response.status}"
        except HTTPError as error:
            if method == "HEAD" and error.code in {405, 501}:
                continue
            return f"HTTP {error.code}"
        except URLError as error:
            return str(error.reason)
        except (OSError, ValueError, UnicodeError) as error:
            return str(error)
    return "HTTP endpoint rejected both HEAD and GET"


def verify_external_links(files: list[Path], timeout: float) -> tuple[list[str], int, int]:
    failures: list[str] = []
    checked = 0
    loopback = 0
    for url, locations in external_references(files).items():
        parsed = urlsplit(url)
        if parsed.hostname in LOOPBACK_HOSTS:
            loopback += 1
            continue
        checked += 1
        error = (
            verify_git_endpoint(url, timeout)
            if is_git_endpoint(url)
            else verify_http_url(url, timeout)
        )
        if error:
            rendered_locations = ", ".join(
                f"{source.relative_to(REPO_ROOT)}:{line_number}"
                for source, line_number in locations
            )
            failures.append(f"{rendered_locations}: unreachable {url!r}: {error}")
    return failures, checked, loopback


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--external",
        action="store_true",
        help="also request public HTTP(S) links and probe HTTPS .git endpoints",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="per-link timeout in seconds when --external is enabled (default: 15)",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    failures: list[str] = []
    checked = 0
    files = document_files()
    anchor_cache: dict[Path, set[str]] = {}
    for source in files:
        text = source.read_text(encoding="utf-8", errors="replace")
        lines = (
            visible_lines(text)
            if source.suffix.lower() in MARKDOWN_EXTENSIONS
            else enumerate(text.splitlines(), 1)
        )
        for line_number, line in lines:
            for match in LINK_PATTERN.finditer(line):
                raw = match.group("markdown") or match.group("html") or ""
                target = link_target(raw)
                resolved = local_target(source, target)
                if resolved is None:
                    continue
                path, fragment = resolved
                checked += 1
                resolved_path = path.resolve()
                try:
                    resolved_path.relative_to(REPO_ROOT)
                except ValueError:
                    relative_source = source.relative_to(REPO_ROOT)
                    failures.append(
                        f"{relative_source}:{line_number}: local target escapes "
                        f"the repository: {target!r}"
                    )
                    continue
                if not resolved_path.exists():
                    relative_source = source.relative_to(REPO_ROOT)
                    failures.append(
                        f"{relative_source}:{line_number}: missing local target "
                        f"{target!r}"
                    )
                    continue
                if fragment and resolved_path.is_file():
                    if resolved_path not in anchor_cache:
                        anchor_cache[resolved_path] = anchors(resolved_path)
                    if fragment not in anchor_cache[resolved_path]:
                        relative_source = source.relative_to(REPO_ROOT)
                        failures.append(
                            f"{relative_source}:{line_number}: missing anchor "
                            f"#{fragment!s} in {target!r}"
                        )

    external_checked = 0
    loopback = 0
    if arguments.external:
        external_failures, external_checked, loopback = verify_external_links(
            external_source_files(), arguments.timeout
        )
        failures.extend(external_failures)

    if failures:
        print("Documentation link check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    message = f"doc_links_ok files={len(files)} local_targets={checked}"
    if arguments.external:
        message += f" external_targets={external_checked} loopback_targets={loopback}"
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
