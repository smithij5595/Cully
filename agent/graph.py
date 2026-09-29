"""Orchestrate Gemini scoring followed by deterministic batch checks."""
from typing import Callable, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from .scoring import PhotoAssessment, PhotoScore, score_photo
from .tools import assign_bucket, find_duplicate_groups, pick_best_of_group


class GraphState(TypedDict):
    image_paths: list[str]
    preferences: str
    scores: dict[str, PhotoAssessment | PhotoScore]
    failures: dict[str, str]
    duplicate_groups: list[list[str]]


def score_node(state: GraphState, config: RunnableConfig) -> dict:
    """Collect Gemini's judgment for each image, retaining per-image failures."""
    scores = {}
    failures = {}
    configurable = config.get("configurable", {})
    on_progress = configurable.get("on_progress")
    total = len(state["image_paths"])

    for completed, image_path in enumerate(state["image_paths"], start=1):
        try:
            scores[image_path] = score_photo(image_path, state["preferences"])
        except Exception as error:
            failures[image_path] = str(error)
        if callable(on_progress):
            on_progress(completed, total, image_path)

    return {"scores": scores, "failures": failures}


def check_node(state: GraphState) -> dict:
    """Assign buckets, then reject weaker copies within duplicate groups."""
    scores = {
        path: PhotoScore(
            **assessment.model_dump(),
            recommendation=assign_bucket(assessment),
        )
        for path, assessment in state["scores"].items()
    }
    image_paths = list(scores)
    duplicate_groups = find_duplicate_groups(image_paths)
    score_values = {path: score.model_dump() for path, score in scores.items()}

    for group in duplicate_groups:
        best_path = pick_best_of_group(group, score_values)
        for image_path in group:
            if image_path == best_path:
                continue
            score = scores[image_path]
            scores[image_path] = score.model_copy(
                update={
                    "recommendation": "reject",
                    "reasoning": (
                        f"Near-duplicate of {best_path}; the stronger image in "
                        "this group was selected. "
                        f"Original assessment: {score.reasoning}"
                    ),
                }
            )

    return {"scores": scores, "duplicate_groups": duplicate_groups}


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("score", score_node)
    graph.add_node("check", check_node)
    graph.set_entry_point("score")
    graph.add_edge("score", "check")
    graph.add_edge("check", END)
    return graph.compile()


def run_batch_pipeline(
    image_paths: list[str],
    preferences: str,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> GraphState:
    """Score a batch, then apply deterministic duplicate handling."""
    if not image_paths:
        raise ValueError("Cannot run the culling pipeline without images")

    app = build_graph()
    config = {"configurable": {"on_progress": on_progress}} if on_progress else None
    return app.invoke(
        {
            "image_paths": image_paths,
            "preferences": preferences,
            "scores": {},
            "failures": {},
            "duplicate_groups": [],
        },
        config=config,
    )


def run_pipeline(image_path: str, preferences: str) -> PhotoScore:
    """Run one image through the batch graph and return its final score."""
    result = run_batch_pipeline([image_path], preferences)
    if image_path in result["failures"]:
        raise RuntimeError(result["failures"][image_path])
    return result["scores"][image_path]
