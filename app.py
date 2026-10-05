"""
Streamlit front end for Cully: the photo culling agent.

Run locally with:  streamlit run app.py

Plans to deploy on Streamlit Community Cloud later. This same file is
the entry point -- no changes needed for hosting.
"""
from datetime import datetime
from io import BytesIO
from pathlib import Path
import base64
import hashlib
import random
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image
import streamlit as st

from agent.graph import run_batch_pipeline, run_pipeline
from agent.tracing import init_tracing

# Tracing must be set up before any agent calls happen.
@st.cache_resource
def setup_tracing():
    return init_tracing()


setup_tracing()

st.set_page_config(page_title="Cully", layout="wide")
st.markdown(
    '<header class="app-header"><span class="app-header__icon" '
    'aria-hidden="true">📸</span><h1>Cully: The Photo Culling Agent</h1></header>',
    unsafe_allow_html=True,
)

LABELS = ["keep", "review", "reject"]
DISPLAY_LABELS = ["review", "reject"]
GALLERY_PAGE_SIZE = 12
THUMBNAIL_SIZE = (640, 480)

st.markdown(
    """
    <style>
    .app-header {
        box-sizing: border-box;
        display: flex;
        align-items: center;
        gap: 1rem;
        width: calc(100% + 10rem);
        height: 76px;
        /*margin: 0 0 1.5rem -5rem;*/
        padding: 0 1.5rem;
        color: #f7f7f2;
        background: #265f35;
        border-bottom: 4px solid #f2c14e;
        box-shadow: 0 2px 8px rgba(24, 14, 34, 0.22);
    }
    header[data-testid="stHeader"] {
        z-index: 999991;
        background: transparent;
    }
    header[data-testid="stHeader"] button {
        color: #f2c14e !important;
    }
    header[data-testid="stHeader"] button[data-testid="stExpandSidebarButton"] {
        color: #f2c14e !important;
        position: absolute;
        top: 184px;
        left: 16px;
        background: transparent !important;
        border: 1px solid #f2c14e !important;
        border-radius: 4px;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
        position: relative;
        z-index: 999992;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button {
        color: #f2c14e !important;
        background: transparent !important;
        border: 1px solid #f2c14e !important;
        border-radius: 4px;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        padding-top: 20px;
    }
    .app-header__icon {
        flex: 0 0 auto;
        font-size: 2rem;
    }
    .app-header h1 {
        margin: 0;
        color: inherit;
        font-size: 1.75rem;
        font-weight: 650;
        letter-spacing: 0;
        line-height: 1.2;
    }
    @media (max-width: 640px) {
        .app-header {
            width: calc(100% + 2rem);
            margin-left: -1rem;
            gap: 0.75rem;
            padding: 0 1rem;
        }
        .app-header h1 {
            font-size: 1.35rem;
        }
    }
    .st-key-keepers_panel {
        background: #f0f2f4;
        border: 1px solid #d7dade;
        border-radius: 6px;
        box-shadow: inset 0 2px 6px rgba(20, 28, 36, 0.12);
        padding: 1rem 1.25rem;
    }
    .photo-thumbnail {
        display: block;
        width: 100%;
        aspect-ratio: 4 / 3;
        overflow: hidden;
        cursor: zoom-in;
    }
    .photo-thumbnail img {
        display: block;
        width: 100%;
        height: auto;
        aspect-ratio: 4 / 3;
        object-fit: contain;
        border-radius: 4px;
    }
    div[data-testid="stDialog"], div[role="dialog"], div[role="dialog"] * {
        animation-duration: 80ms !important;
        transition-duration: 80ms !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- session state ---
if "batches" not in st.session_state:
    st.session_state.batches = {}
if "active_batch_id" not in st.session_state:
    st.session_state.active_batch_id = None
if "upload_draft_id" not in st.session_state:
    st.session_state.upload_draft_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def create_batch(name: str, preferences: str) -> dict:
    batch_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return {
        "id": batch_id,
        "name": name.strip() or f"Batch {len(st.session_state.batches) + 1}",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "preferences": preferences,
        "containers": {label: [] for label in LABELS},
        "failed": [],
        "original_names": {},
        "status": "scoring",
        "celebrated": False,
    }


def keeper_zip(batch: dict) -> bytes:
    """Build a ZIP from the original uploaded keeper files."""
    output = BytesIO()
    used_names = set()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for path in batch["containers"]["keep"]:
            original_name = batch["original_names"].get(path, Path(path).name)
            archive_name = original_name
            stem, suffix = Path(original_name).stem, Path(original_name).suffix
            index = 2
            while archive_name in used_names:
                archive_name = f"{stem}_{index}{suffix}"
                index += 1
            used_names.add(archive_name)
            archive.write(path, archive_name)
    return output.getvalue()


@st.cache_data
def render_thumbnail(path: str) -> bytes:
    """Create a fixed-size, centered thumbnail without applying EXIF rotation."""
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", THUMBNAIL_SIZE, (28, 31, 36))
        offset = (
            (THUMBNAIL_SIZE[0] - image.width) // 2,
            (THUMBNAIL_SIZE[1] - image.height) // 2,
        )
        canvas.paste(image, offset)
        output = BytesIO()
        canvas.save(output, format="JPEG", quality=88, optimize=True)
    return output.getvalue()


def render_clickable_thumbnail(path: str) -> None:
    thumbnail = base64.b64encode(render_thumbnail(path)).decode("ascii")
    preview_key = f"photo_preview_{hashlib.sha256(path.encode('utf-8')).hexdigest()}"
    st.markdown(
        f"""
        <style>
        .st-key-{preview_key} button {{
            width: 100%;
            height: auto;
            min-height: 0;
            aspect-ratio: 4 / 3;
            padding: 0;
            color: transparent;
            font-size: 0;
            background-color: #1c1f24;
            background-image: url("data:image/jpeg;base64,{thumbnail}");
            background-position: center;
            background-repeat: no-repeat;
            background-size: contain;
            border: 0;
            border-radius: 4px;
        }}
        .st-key-{preview_key} button p {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    if st.button(
        "Open photo preview",
        key=preview_key,
        help="View photo",
        width="stretch",
    ):
        st.session_state.preview_path = path


def start_new_batch() -> None:
    current_upload_key = (
        f"batch_uploads_{st.session_state.active_batch_id}"
        if st.session_state.active_batch_id
        else f"batch_uploads_{st.session_state.upload_draft_id}"
    )
    st.session_state.pop(current_upload_key, None)
    st.session_state.upload_draft_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    st.session_state.active_batch_id = None
    st.session_state.pop("preview_path", None)
    for key in list(st.session_state):
        if key.startswith("batch_selector_"):
            st.session_state.pop(key, None)
    st.session_state.pop("batch_uploads", None)


def selection_key(batch: dict, label: str, path: str) -> str:
    return f"selected_{batch['id']}_{label}_{path}"


def clear_bucket_selection(batch: dict, label: str) -> None:
    for path in batch["containers"][label]:
        st.session_state.pop(selection_key(batch, label, path), None)


def select_all_in_bucket(batch: dict, label: str) -> None:
    for path in batch["containers"][label]:
        st.session_state[selection_key(batch, label, path)] = True


def clear_all_bucket_selection(batch: dict, label: str) -> None:
    clear_bucket_selection(batch, label)


def clear_photo_preview() -> None:
    st.session_state.pop("preview_path", None)


@st.dialog("Photo preview", on_dismiss=clear_photo_preview)
def show_photo(path: str):
    image_data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    mime_type = "image/png" if Path(path).suffix.lower() == ".png" else "image/jpeg"
    st.markdown(
        f'<img src="data:{mime_type};base64,{image_data}" alt="Photo preview" '
        'style="display:block;max-width:100%;max-height:75vh;height:auto;'
        'object-fit:contain;margin:auto;">',
        unsafe_allow_html=True,
    )


def draw_photo(batch: dict, col, label: str, path: str):
    """Draw a clickable thumbnail with icon-only move actions."""
    with col:
        st.checkbox(
            "Select photo",
            key=selection_key(batch, label, path),
            label_visibility="collapsed",
        )
        render_clickable_thumbnail(path)
        if label == "review":
            move_actions = [
                ("keep", ":material/drive_file_move:", "Move to Keepers"),
                ("reject", ":material/delete:", "Reject photo"),
            ]
        elif label == "keep":
            move_actions = [("reject", ":material/delete:", "Reject photo")]
        else:
            move_actions = [("keep", ":material/drive_file_move:", "Move to Keepers")]

        action_columns = st.columns(len(move_actions))
        for action_column, (move_to, icon, action) in zip(action_columns, move_actions):
            with action_column:
                if st.button(
                    "",
                    icon=icon,
                    help=action,
                    key=f"move_{path}_{label}_{move_to}",
                    width="stretch",
                ):
                    batch["containers"][label].remove(path)
                    batch["containers"][move_to].insert(0, path)
                    st.session_state.pop(selection_key(batch, label, path), None)
                    st.rerun()


def render_gallery(batch: dict, label: str, container) -> None:
    """Render one bucket as a bounded, newest-first thumbnail gallery."""
    paths = batch["containers"][label]
    page_count = max(1, (len(paths) + GALLERY_PAGE_SIZE - 1) // GALLERY_PAGE_SIZE)
    page_key = f"gallery_page_{batch['id']}_{label}"
    page = min(st.session_state.get(page_key, 0), page_count - 1)
    st.session_state[page_key] = page

    with container:
        selected_paths = [
            path
            for path in paths
            if st.session_state.get(selection_key(batch, label, path), False)
        ]
        toolbar_left, toolbar_mid, toolbar_right = st.columns([1, 1, 2])
        with toolbar_left:
            st.button(
                "Select all",
                key=f"select_all_{batch['id']}_{label}",
                on_click=select_all_in_bucket,
                args=(batch, label),
                disabled=not paths,
                width="stretch",
            )
        with toolbar_mid:
            st.button(
                "Clear",
                key=f"clear_all_{batch['id']}_{label}",
                on_click=clear_all_bucket_selection,
                args=(batch, label),
                disabled=not selected_paths,
                width="stretch",
            )
        with toolbar_right:
            move_options = [target for target in LABELS if target != label]
            bulk_target = st.selectbox(
                "Move selected to",
                move_options,
                key=f"bulk_target_{batch['id']}_{label}",
                disabled=not selected_paths,
            )
            if st.button(
                f"Move {len(selected_paths)} selected",
                key=f"bulk_move_{batch['id']}_{label}",
                disabled=not selected_paths,
                width="stretch",
            ):
                for path in selected_paths:
                    batch["containers"][label].remove(path)
                    batch["containers"][bulk_target].insert(0, path)
                    st.session_state.pop(selection_key(batch, label, path), None)
                st.rerun()

        if paths:
            nav_left, page_label, nav_right = st.columns([1, 2, 1])
            with nav_left:
                if st.button(
                    "",
                    icon=":material/chevron_left:",
                    help="Previous page",
                    key=f"prev_{batch['id']}_{label}",
                    disabled=page == 0,
                    width="stretch",
                ):
                    st.session_state[page_key] = page - 1
                    st.rerun()
            with page_label:
                st.caption(f"Page {page + 1} of {page_count} · {len(paths)} photos")
            with nav_right:
                if st.button(
                    "",
                    icon=":material/chevron_right:",
                    help="Next page",
                    key=f"next_{batch['id']}_{label}",
                    disabled=page >= page_count - 1,
                    width="stretch",
                ):
                    st.session_state[page_key] = page + 1
                    st.rerun()

            start = page * GALLERY_PAGE_SIZE
            for offset in range(0, min(GALLERY_PAGE_SIZE, len(paths) - start), 3):
                row = st.columns(3)
                for column, path in zip(row, paths[start + offset:start + offset + 3]):
                    draw_photo(batch, column, label, path)
        else:
            st.caption("Nothing here yet.")


# --- batch navigation ---
with st.sidebar:
    st.subheader("Batches")
    if st.session_state.batches:
        batch_ids = list(st.session_state.batches)
        selector_options = [None, *batch_ids]
        selector_key = f"batch_selector_{st.session_state.active_batch_id or 'new'}"
        selected_id = st.selectbox(
            "Open batch",
            selector_options,
            index=selector_options.index(st.session_state.active_batch_id),
            format_func=lambda batch_id: (
                "New batch"
                if batch_id is None
                else (
                f"{st.session_state.batches[batch_id]['name']} · "
                f"{st.session_state.batches[batch_id]['created_at']}"
                )
            ),
            key=selector_key,
        )
        if selected_id != st.session_state.active_batch_id:
            current_upload_key = (
                f"batch_uploads_{st.session_state.active_batch_id}"
                if st.session_state.active_batch_id
                else f"batch_uploads_{st.session_state.upload_draft_id}"
            )
            st.session_state.pop(current_upload_key, None)
            st.session_state.upload_draft_id = datetime.now().strftime(
                "%Y%m%d-%H%M%S-%f"
            )
            st.session_state.active_batch_id = selected_id
            st.session_state.pop("preview_path", None)
            st.rerun()

    st.button(
        "＋ New batch",
        width="stretch",
        on_click=start_new_batch,
    )


active_batch = (
    st.session_state.batches.get(st.session_state.active_batch_id)
    if st.session_state.active_batch_id
    else None
)
batch_complete = active_batch is not None and active_batch["status"] == "complete"

# --- input ---
preferences = st.text_input(
    "What are you shooting for creatively?",
    placeholder="e.g. sharp action shots, landscapes and artistic shots of plants, pets",
    value=active_batch["preferences"] if active_batch else "",
    disabled=active_batch is not None,
)
batch_name = st.text_input(
    "Batch name",
    placeholder="e.g. SF Marathon",
    value=active_batch["name"] if active_batch else "",
    disabled=active_batch is not None,
)
uploader_key = (
    f"batch_uploads_{active_batch['id']}"
    if active_batch
    else f"batch_uploads_{st.session_state.upload_draft_id}"
)
uploaded_files = st.file_uploader(
    "Upload photos",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    disabled=active_batch is not None,
    key=uploader_key,
)
if active_batch:
    original_names = list(dict.fromkeys(active_batch["original_names"].values()))
    if original_names:
        with st.expander(f"Photos in this batch ({len(original_names)})"):
            for original_name in original_names:
                st.write(original_name)
if uploaded_files:
    seen = set()
    unique_files = []
    for file in uploaded_files:
        digest = hashlib.sha256(file.getvalue()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            unique_files.append((file, digest))
    uploaded_files = unique_files

run_clicked = bool(uploaded_files) and st.button(
    "Run Cully", disabled=active_batch is not None
)

if run_clicked:
    active_batch = create_batch(batch_name, preferences)
    st.session_state.batches[active_batch["id"]] = active_batch
    st.session_state.active_batch_id = active_batch["id"]

# --- scoring progress ---
progress_messages = [
    "Thinking hard about whether life imitates art or art imitates life...",
    "Consulting the tiny gallery curator in my head...",
    "Looking for the decisive moment hiding in the frame...",
    "Comparing composition, focus, and vibes with great seriousness...",
    "Giving this photo the thoughtful squint it deserves...",
    "What really is art....?",
    "Having a quick snack. What? I can multi-task...",
]

# --- completion summary ---
if batch_complete:
    if not active_batch["celebrated"]:
        st.toast("Your batch is ready!", icon="📸")
        active_batch["celebrated"] = True
    counts = {
        label: len(active_batch["containers"][label]) for label in LABELS
    }
    st.success(
        f"Culling complete: {sum(counts.values())} images reviewed, "
        f"{counts['keep']} keeper(s), {counts['review']} you should review, "
        f"and {counts['reject']} reject(s)."
    )
    if counts["review"]:
        st.info(
            f"Start with the {counts['review']} photo(s) in Review. "
            "Trust your eye, move the ones you want to Keepers, then you can download all of your keepers."
        )
    else:
        st.info("No photos need a second look! Your keepers are ready below.")

    with st.container(border=True, key="keepers_panel"):
        st.subheader(f"Keepers ({counts['keep']})")
        render_gallery(active_batch, "keep", st.container())

        st.subheader("Batch actions")
        keeper_count = len(active_batch["containers"]["keep"])
        st.write(f"{keeper_count} keeper(s) ready to export.")
        if keeper_count:
            st.download_button(
                "Download Keepers",
                data=keeper_zip(active_batch),
                file_name=f"{active_batch['name'].lower().replace(' ', '-')}-keepers.zip",
                mime="application/zip",
                width="stretch",
            )
        st.caption("The ZIP contains the original uploaded files.")

if run_clicked:
    upload_dir = Path("uploaded_photos")
    upload_dir.mkdir(exist_ok=True)

    image_paths = []
    for file, digest in uploaded_files:
        original_name = Path(file.name)
        safe_name = f"{original_name.stem}_{digest[:12]}{original_name.suffix.lower()}"
        save_path = upload_dir / safe_name
        save_path.write_bytes(file.getvalue())
        image_path = str(save_path)
        image_paths.append(image_path)
        active_batch["original_names"][image_path] = file.name

    progress = st.progress(0.0, text="Scoring photos...")
    def update_scoring_progress(completed: int, total: int, image_path: str) -> None:
        phrase = random.choice(progress_messages)
        progress.progress(
            completed / total,
            text=f"{phrase} ({completed} of {total})",
        )

    result = run_batch_pipeline(
        image_paths,
        preferences,
        on_progress=update_scoring_progress,
    )
    for image_path in image_paths:
        if image_path in result["failures"]:
            active_batch["failed"].append(
                {
                    "path": image_path,
                    "preferences": preferences,
                    "error": result["failures"][image_path],
                }
            )
            continue
        score = result["scores"][image_path]
        active_batch["containers"][score.recommendation].insert(0, image_path)

    progress.progress(1.0, text="All done. Your gallery is ready.")
    active_batch["status"] = "complete"
    st.rerun()

# --- completed galleries ---
if active_batch and not run_clicked:
    for label in DISPLAY_LABELS:
        st.divider()
        st.subheader(label.capitalize())
        render_gallery(active_batch, label, st.container())

# --- needs-retry section ---
if active_batch and active_batch["failed"]:
    st.divider()
    st.subheader(f"⚠️ Needs retry ({len(active_batch['failed'])})")
    for item in list(active_batch["failed"]):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.image(item["path"], width=150)
            st.caption(f"Failed: {item['error']}")
        with c2:
            if st.button("Retry", key=f"retry_{item['path']}"):
                try:
                    score = run_pipeline(item["path"], item["preferences"])
                    active_batch["containers"][score.recommendation].insert(0, item["path"])
                    active_batch["failed"].remove(item)
                except Exception as e:
                    item["error"] = str(e)
                st.rerun()

preview_path = st.session_state.get("preview_path")
if active_batch and preview_path:
    batch_paths = {
        path
        for paths in active_batch["containers"].values()
        for path in paths
    }
    if preview_path in batch_paths:
        show_photo(preview_path)
    else:
        clear_photo_preview()
