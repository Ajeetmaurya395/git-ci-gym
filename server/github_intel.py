"""
GitHub PR Intelligence — CodeRabbit-style context fetching.

Fetches rich PR context from the GitHub REST API:
  - PR metadata (title, author, labels, branches)
  - Changed files with diff hunks
  - CI check runs and failure logs
  - Review comments (including bot reviews like CodeRabbit)
  - Security scan summary

Works without authentication for public repos (60 req/hr limit).
Set GITHUB_TOKEN env var for private repos and higher limits (5000 req/hr).
"""

from __future__ import annotations

import os
import re
import logging
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

# ─── Data Models ──────────────────────────────────────────────────────


@dataclass
class ChangedFile:
    """A file changed in the PR."""
    filename: str
    status: str  # added, removed, modified, renamed
    additions: int = 0
    deletions: int = 0
    patch: str = ""  # diff hunk (truncated)

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "status": self.status,
            "additions": self.additions,
            "deletions": self.deletions,
            "patch": self.patch[:2000],  # truncate large patches
        }


@dataclass
class CheckRun:
    """A CI check run (e.g., GitHub Actions job)."""
    name: str
    status: str  # completed, in_progress, queued
    conclusion: str  # success, failure, neutral, cancelled, etc.
    html_url: str = ""
    run_id: int = 0
    job_id: int = 0
    failure_summary: str = ""  # extracted error message

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "conclusion": self.conclusion,
            "html_url": self.html_url,
            "failure_summary": self.failure_summary[:1000],
        }


@dataclass
class ReviewComment:
    """A PR review comment."""
    author: str
    body: str
    state: str  # APPROVED, CHANGES_REQUESTED, COMMENTED
    is_bot: bool = False
    submitted_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "author": self.author,
            "body": self.body[:2000],
            "state": self.state,
            "is_bot": self.is_bot,
            "submitted_at": self.submitted_at,
        }


@dataclass
class SecurityScan:
    """Security scan summary extracted from CI checks and PR content."""
    vulnerabilities: list[str] = field(default_factory=list)
    dependency_issues: list[str] = field(default_factory=list)
    lint_issues: list[str] = field(default_factory=list)
    scan_sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vulnerabilities": self.vulnerabilities[:20],
            "dependency_issues": self.dependency_issues[:20],
            "lint_issues": self.lint_issues[:20],
            "scan_sources": self.scan_sources,
            "total_issues": len(self.vulnerabilities) + len(self.dependency_issues) + len(self.lint_issues),
        }


@dataclass
class PRIntelligence:
    """Complete PR intelligence — all context needed for smart repair."""

    # PR metadata
    pr_number: int = 0
    title: str = ""
    body: str = ""
    author: str = ""
    state: str = ""
    base_branch: str = ""
    head_branch: str = ""
    labels: list[str] = field(default_factory=list)
    html_url: str = ""

    # Changes
    changed_files: list[ChangedFile] = field(default_factory=list)
    total_additions: int = 0
    total_deletions: int = 0
    total_changed_files: int = 0

    # CI
    check_runs: list[CheckRun] = field(default_factory=list)
    failing_checks: int = 0
    passing_checks: int = 0

    # Reviews
    reviews: list[ReviewComment] = field(default_factory=list)

    # Security
    security: SecurityScan = field(default_factory=SecurityScan)

    # Meta
    fetched: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "pr_number": self.pr_number,
            "title": self.title,
            "body": self.body[:3000],
            "author": self.author,
            "state": self.state,
            "base_branch": self.base_branch,
            "head_branch": self.head_branch,
            "labels": self.labels,
            "html_url": self.html_url,
            "changed_files": [f.to_dict() for f in self.changed_files],
            "total_additions": self.total_additions,
            "total_deletions": self.total_deletions,
            "total_changed_files": self.total_changed_files,
            "check_runs": [c.to_dict() for c in self.check_runs],
            "failing_checks": self.failing_checks,
            "passing_checks": self.passing_checks,
            "reviews": [r.to_dict() for r in self.reviews],
            "security": self.security.to_dict(),
            "fetched": self.fetched,
            "error": self.error,
        }

    def to_agent_context(self) -> str:
        """Format PR intelligence as concise context for the AI agent's prompt."""
        if not self.fetched:
            return ""

        lines = [
            f"## PR Context (GitHub API)",
            f"**PR #{self.pr_number}**: {self.title}",
            f"**Author**: {self.author}  |  **Base**: {self.base_branch} ← **Head**: {self.head_branch}",
            f"**Changed Files**: {self.total_changed_files} ({self.total_additions}+ / {self.total_deletions}-)",
        ]

        if self.labels:
            lines.append(f"**Labels**: {', '.join(self.labels)}")

        # CI failures
        failing = [c for c in self.check_runs if c.conclusion == "failure"]
        passing = [c for c in self.check_runs if c.conclusion == "success"]
        if failing:
            lines.append(f"\n### CI Failures ({len(failing)} failing, {len(passing)} passing)")
            for check in failing:
                lines.append(f"- ❌ **{check.name}**")
                if check.failure_summary:
                    # Indent error details
                    for err_line in check.failure_summary.split("\n")[:10]:
                        lines.append(f"  > {err_line}")
        elif passing:
            lines.append(f"\n### CI: All {len(passing)} checks passing ✅")

        # Key changed files
        if self.changed_files:
            lines.append(f"\n### Changed Files (top 15)")
            for cf in self.changed_files[:15]:
                icon = {"added": "🟢", "removed": "🔴", "modified": "🟡", "renamed": "🔄"}.get(cf.status, "📄")
                lines.append(f"- {icon} `{cf.filename}` (+{cf.additions}/-{cf.deletions})")

        # Review comments
        meaningful_reviews = [r for r in self.reviews if r.state != "COMMENTED" or len(r.body) > 50]
        if meaningful_reviews:
            lines.append(f"\n### Review Comments ({len(meaningful_reviews)})")
            for review in meaningful_reviews[:5]:
                tag = "🤖" if review.is_bot else "👤"
                lines.append(f"- {tag} **{review.author}** ({review.state}): {review.body[:300]}")

        # Security
        total_sec = self.security.to_dict()["total_issues"]
        if total_sec > 0:
            lines.append(f"\n### Security Scan ({total_sec} issues)")
            for v in self.security.vulnerabilities[:5]:
                lines.append(f"- 🔴 VULN: {v}")
            for d in self.security.dependency_issues[:5]:
                lines.append(f"- 🟡 DEP: {d}")
            for l in self.security.lint_issues[:5]:
                lines.append(f"- 🔵 LINT: {l}")

        return "\n".join(lines)


# ─── GitHub API Client ────────────────────────────────────────────────


def _headers() -> dict[str, str]:
    """Build request headers with optional auth."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _get(url: str, accept: str | None = None, timeout: int = 15) -> requests.Response | None:
    """Make a GET request to the GitHub API with error handling."""
    headers = _headers()
    if accept:
        headers["Accept"] = accept
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code == 403:
            logger.warning("GitHub API rate limit reached. Set GITHUB_TOKEN for higher limits.")
            return None
        if resp.status_code >= 400:
            logger.warning(f"GitHub API error {resp.status_code}: {url}")
            return None
        return resp
    except requests.RequestException as e:
        logger.warning(f"GitHub API request failed: {e}")
        return None


def _parse_pr_url(url: str) -> tuple[str, str, int] | None:
    """Extract (owner, repo, pr_number) from a GitHub PR URL."""
    parsed = urlparse(url)
    if parsed.netloc != "github.com":
        return None
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 4 and parts[2] == "pull" and parts[3].isdigit():
        return parts[0], parts[1], int(parts[3])
    return None


# ─── Fetchers ─────────────────────────────────────────────────────────


def _fetch_pr_metadata(owner: str, repo: str, pr_number: int) -> dict[str, Any] | None:
    """Fetch PR metadata."""
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
    resp = _get(url)
    if resp is None:
        return None
    return resp.json()


def _fetch_changed_files(owner: str, repo: str, pr_number: int) -> list[ChangedFile]:
    """Fetch list of changed files with patches."""
    files = []
    page = 1
    while True:
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files?per_page=100&page={page}"
        resp = _get(url)
        if resp is None:
            break
        data = resp.json()
        if not data:
            break
        for f in data:
            files.append(ChangedFile(
                filename=f.get("filename", ""),
                status=f.get("status", "modified"),
                additions=f.get("additions", 0),
                deletions=f.get("deletions", 0),
                patch=f.get("patch", "")[:2000],
            ))
        if len(data) < 100:
            break
        page += 1
    return files


def _fetch_check_runs(owner: str, repo: str, head_sha: str) -> list[CheckRun]:
    """Fetch CI check runs for the PR's head commit."""
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{head_sha}/check-runs?per_page=100"
    resp = _get(url)
    if resp is None:
        return []
    data = resp.json()
    checks = []
    for cr in data.get("check_runs", []):
        check = CheckRun(
            name=cr.get("name", ""),
            status=cr.get("status", ""),
            conclusion=cr.get("conclusion", "") or "",
            html_url=cr.get("html_url", ""),
        )
        # Extract failure summary from the check output
        output = cr.get("output", {}) or {}
        if check.conclusion == "failure":
            summary = output.get("summary", "") or ""
            text = output.get("text", "") or ""
            check.failure_summary = (summary + "\n" + text).strip()[:1500]

        # Try to get job_id for log fetching
        details_url = cr.get("details_url", "")
        if "/runs/" in details_url:
            try:
                check.run_id = int(details_url.split("/runs/")[-1].split("/")[0])
            except (ValueError, IndexError):
                pass

        checks.append(check)
    return checks


def _fetch_action_logs(owner: str, repo: str, check: CheckRun) -> str:
    """Fetch GitHub Actions job logs for a failing check."""
    if not check.failure_summary and check.conclusion == "failure":
        # Try to get logs via the Actions API
        # First find the workflow run
        url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs?per_page=5&status=failure"
        resp = _get(url)
        if resp is None:
            return ""
        runs = resp.json().get("workflow_runs", [])
        for run in runs:
            jobs_url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run['id']}/jobs"
            jobs_resp = _get(jobs_url)
            if jobs_resp is None:
                continue
            for job in jobs_resp.json().get("jobs", []):
                if job.get("name", "") == check.name and job.get("conclusion") == "failure":
                    # Extract failure info from steps
                    failed_steps = [
                        s for s in job.get("steps", [])
                        if s.get("conclusion") == "failure"
                    ]
                    if failed_steps:
                        return "\n".join(
                            f"Step '{s['name']}' failed ({s.get('conclusion', 'unknown')})"
                            for s in failed_steps
                        )
    return ""


def _fetch_reviews(owner: str, repo: str, pr_number: int) -> list[ReviewComment]:
    """Fetch PR review comments."""
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/reviews?per_page=100"
    resp = _get(url)
    if resp is None:
        return []
    reviews = []
    for r in resp.json():
        author = r.get("user", {}).get("login", "unknown")
        reviews.append(ReviewComment(
            author=author,
            body=r.get("body", "") or "",
            state=r.get("state", "COMMENTED"),
            is_bot=r.get("user", {}).get("type", "") == "Bot",
            submitted_at=r.get("submitted_at", ""),
        ))
    return reviews


def _build_security_scan(
    check_runs: list[CheckRun],
    changed_files: list[ChangedFile],
    reviews: list[ReviewComment],
) -> SecurityScan:
    """Build a security scan summary from available data."""
    scan = SecurityScan()

    # Analyze check runs for security-related checks
    security_keywords = {"security", "scan", "audit", "snyk", "dependabot", "codeql", "semgrep", "trivy", "bandit"}
    lint_keywords = {"lint", "ruff", "flake8", "eslint", "pylint", "mypy", "clippy", "format", "prettier"}
    dep_keywords = {"dependency", "dependabot", "renovate", "npm audit", "pip-audit"}

    for check in check_runs:
        name_lower = check.name.lower()

        # Categorize by check name
        if any(kw in name_lower for kw in security_keywords):
            scan.scan_sources.append(check.name)
            if check.conclusion == "failure":
                msg = check.failure_summary or f"{check.name} failed"
                scan.vulnerabilities.append(msg[:300])

        elif any(kw in name_lower for kw in lint_keywords):
            scan.scan_sources.append(check.name)
            if check.conclusion == "failure":
                msg = check.failure_summary or f"{check.name} failed"
                scan.lint_issues.append(msg[:300])

        elif any(kw in name_lower for kw in dep_keywords):
            scan.scan_sources.append(check.name)
            if check.conclusion == "failure":
                msg = check.failure_summary or f"{check.name} failed"
                scan.dependency_issues.append(msg[:300])

    # Scan changed files for common security patterns
    sensitive_patterns = {
        r"\.env$": "Environment file modified",
        r"secret|password|api_key|token": "Potential secret exposure in filename",
        r"\.pem$|\.key$|\.cert$": "Certificate/key file modified",
        r"requirements.*\.txt$|Cargo\.toml$|package\.json$": "Dependency file modified",
        r"Dockerfile|docker-compose": "Container config modified",
    }
    for cf in changed_files:
        for pattern, description in sensitive_patterns.items():
            if re.search(pattern, cf.filename, re.IGNORECASE):
                scan.dependency_issues.append(f"{description}: `{cf.filename}`")
                break

    # Extract security mentions from reviews
    for review in reviews:
        body_lower = review.body.lower()
        if any(kw in body_lower for kw in ["security", "vulnerability", "xss", "injection", "auth", "credential"]):
            scan.vulnerabilities.append(f"Review by {review.author}: {review.body[:200]}")

    # Deduplicate
    scan.vulnerabilities = list(dict.fromkeys(scan.vulnerabilities))
    scan.dependency_issues = list(dict.fromkeys(scan.dependency_issues))
    scan.lint_issues = list(dict.fromkeys(scan.lint_issues))
    scan.scan_sources = list(dict.fromkeys(scan.scan_sources))

    return scan


# ─── Public API ───────────────────────────────────────────────────────


def fetch_pr_intelligence(pr_url: str) -> PRIntelligence:
    """
    Fetch complete PR intelligence from the GitHub API.

    Args:
        pr_url: GitHub PR URL (e.g., https://github.com/owner/repo/pull/123)

    Returns:
        PRIntelligence with all available context, or an error state.
    """
    intel = PRIntelligence()

    parsed = _parse_pr_url(pr_url)
    if not parsed:
        intel.error = f"Not a valid GitHub PR URL: {pr_url}"
        return intel

    owner, repo, pr_number = parsed
    intel.pr_number = pr_number
    intel.html_url = pr_url

    # 1. PR metadata
    meta = _fetch_pr_metadata(owner, repo, pr_number)
    if meta is None:
        intel.error = "Failed to fetch PR metadata. GitHub API may be rate-limited."
        return intel

    intel.title = meta.get("title", "")
    intel.body = meta.get("body", "") or ""
    intel.author = meta.get("user", {}).get("login", "unknown")
    intel.state = meta.get("state", "")
    intel.base_branch = meta.get("base", {}).get("ref", "")
    intel.head_branch = meta.get("head", {}).get("ref", "")
    intel.labels = [l.get("name", "") for l in meta.get("labels", [])]
    head_sha = meta.get("head", {}).get("sha", "")

    # 2. Changed files
    intel.changed_files = _fetch_changed_files(owner, repo, pr_number)
    intel.total_changed_files = len(intel.changed_files)
    intel.total_additions = sum(f.additions for f in intel.changed_files)
    intel.total_deletions = sum(f.deletions for f in intel.changed_files)

    # 3. CI check runs
    if head_sha:
        intel.check_runs = _fetch_check_runs(owner, repo, head_sha)
        # Try to enrich failing checks with action logs
        for check in intel.check_runs:
            if check.conclusion == "failure" and not check.failure_summary:
                extra_log = _fetch_action_logs(owner, repo, check)
                if extra_log:
                    check.failure_summary = extra_log
        intel.failing_checks = sum(1 for c in intel.check_runs if c.conclusion == "failure")
        intel.passing_checks = sum(1 for c in intel.check_runs if c.conclusion == "success")

    # 4. Reviews
    intel.reviews = _fetch_reviews(owner, repo, pr_number)

    # 5. Security scan
    intel.security = _build_security_scan(intel.check_runs, intel.changed_files, intel.reviews)

    intel.fetched = True
    return intel
