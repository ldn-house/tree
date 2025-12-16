"""
Shared GitHub CI artifact configuration.

Used by flash.py (host) and selfupdate.py (Pico) to access CI artifacts.
"""

# GitHub repository
GITHUB_REPO = "ldn-house/tree"

# GitHub Actions workflow file
GITHUB_WORKFLOW = "build.yml"

# Default branch to fetch from
GITHUB_BRANCH = "main"

# Artifact name from workflow
ARTIFACT_NAME = "firmware"

# Files included in firmware artifact
FIRMWARE_FILES = ["main.py", "ota.py", "mqtt.py", "selfupdate.py", "github.py", "version.py"]


def _url_encode_branch(branch):
    """URL-encode branch name for use in URL path.

    Handles slashes in branch names (e.g., "claude/feature" -> "claude%2Ffeature").
    Works on both CPython and MicroPython.
    """
    # Only encode the slash - other special chars are unlikely in branch names
    return branch.replace("/", "%2F")


def nightly_link_url(branch=GITHUB_BRANCH):
    """Build nightly.link URL for downloading artifacts without authentication.

    nightly.link provides unauthenticated access to GitHub Actions artifacts.
    URL format: https://nightly.link/<owner>/<repo>/workflows/<workflow>/<branch>/<artifact>.zip
    """
    encoded_branch = _url_encode_branch(branch)
    return f"https://nightly.link/{GITHUB_REPO}/workflows/{GITHUB_WORKFLOW}/{encoded_branch}/{ARTIFACT_NAME}.zip"
