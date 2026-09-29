"""
Plain-code tools the agent graph calls. Anything that doesn't need
judgment -- hashing, duplicate grouping, picking a "best of group" --
lives here as a regular function, not an LLM call.

Keeping these separate from scoring.py is "LLM only where
judgment is actually needed"
"""
from PIL import Image
import imagehash
from typing import Literal


def assign_bucket(score: dict | object) -> Literal["keep", "review", "reject"]:
    """Apply the culling thresholds to an image's Gemini assessment."""
    def value(key: str):
        return score[key] if isinstance(score, dict) else getattr(score, key)

    dimensions = ("sharpness", "framing", "expression")
    if any(int(value(dimension)) <= 3 for dimension in dimensions):
        return "reject"
    if all(int(value(dimension)) >= 7 for dimension in dimensions) and value(
        "matches_preferences"
    ):
        return "keep"
    return "review"

def compute_phash(image_path: str) -> imagehash.ImageHash:
    """Perceptual hash of one image, used to detect near-duplicates."""
    with Image.open(image_path) as img:
        return imagehash.phash(img)

def find_duplicate_groups(
    image_paths: list[str], max_distance: int = 5
) -> list[list[str]]:
    """
    Groups images that are likely the same moment (e.g. burst shots).

    Uses connected components, so duplicate relationships are transitive.
    Singleton images are omitted from the result.
    """
    hashes = {path: compute_phash(path) for path in image_paths}
    parent = {path: path for path in image_paths}

    def find(path: str) -> str:
        while parent[path] != path:
            parent[path] = parent[parent[path]]
            path = parent[path]
        return path

    def union(first: str, second: str) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    paths = list(image_paths)
    for index, first in enumerate(paths):
        for second in paths[index + 1 :]:
            if hashes[first] - hashes[second] <= max_distance:
                union(first, second)

    groups: dict[str, list[str]] = {}
    for path in paths:
        groups.setdefault(find(path), []).append(path)
    return [group for group in groups.values() if len(group) > 1]


def pick_best_of_group(group: list[str], scores: dict[str, dict]) -> str:
    """
    Given a duplicate group and each photo's score dict (from
    scoring.score_photo), returns the path of the best one.

    No LLM call here -- just compares scores already computed.

    Scores sharpness most heavily because a technically sharp image is
    usually the strongest choice within a burst.
    """
    if not group:
        raise ValueError("Cannot choose from an empty duplicate group")

    def value(path: str, key: str) -> int:
        score = scores[path]
        return int(score[key] if isinstance(score, dict) else getattr(score, key))

    return max(
        group,
        key=lambda path: (
            value(path, "sharpness") * 2
            + value(path, "framing")
            + value(path, "expression"),
            value(path, "sharpness"),
        ),
    )
