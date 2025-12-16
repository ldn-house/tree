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
FIRMWARE_FILES = ["main.py", "ota.py", "mqtt.py", "selfupdate.py", "github.py"]


def nightly_link_url(branch=GITHUB_BRANCH):
    """Build nightly.link URL for downloading artifacts without authentication.

    nightly.link provides unauthenticated access to GitHub Actions artifacts.
    URL format: https://nightly.link/<owner>/<repo>/workflows/<workflow>/<branch>/<artifact>.zip
    """
    return f"https://nightly.link/{GITHUB_REPO}/workflows/{GITHUB_WORKFLOW}/{branch}/{ARTIFACT_NAME}.zip"
