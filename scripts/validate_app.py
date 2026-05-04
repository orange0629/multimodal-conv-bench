"""
Streamlit app for human validation of DialogVis conversations.

Shows each conversation's turns (with images), final question, MCQ options,
and ground-truth reasoning. The "Keep" button appends the current line to
outputs/conversations/<taxonomy>.kept.jsonl. The "Skip" button advances
without saving. A small status bar shows progress and counts.

Run:
  streamlit run scripts/validate_app.py
  # then in the sidebar pick a taxonomy file
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
CONV_DIR = ROOT / "outputs" / "conversations"
IMG_ROOT = ROOT  # source_path is relative to repo root


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def kept_path_for(src: Path) -> Path:
    return src.with_name(src.stem + ".kept.jsonl")


def load_kept_ids(kept_path: Path) -> set[str]:
    if not kept_path.exists():
        return set()
    ids = set()
    with kept_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["id"])
            except Exception:
                continue
    return ids


def append_kept(kept_path: Path, conv: dict) -> None:
    with kept_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(conv, ensure_ascii=False) + "\n")


def remove_kept(kept_path: Path, conv_id: str) -> None:
    if not kept_path.exists():
        return
    lines = []
    with kept_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                if json.loads(s)["id"] == conv_id:
                    continue
            except Exception:
                pass
            lines.append(s)
    tmp = kept_path.with_suffix(kept_path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    tmp.replace(kept_path)


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

st.set_page_config(page_title="DialogVis Validator", layout="wide")
st.title("DialogVis — Human Validation")

# Sidebar: pick taxonomy
files = sorted(CONV_DIR.glob("*.jsonl"))
files = [f for f in files if not f.name.endswith(".kept.jsonl")]
if not files:
    st.error(f"No conversation files found in {CONV_DIR}")
    st.stop()

file_names = [f.name for f in files]
default_idx = 0
for i, n in enumerate(file_names):
    if "belief_revision" in n:
        default_idx = i
        break

selected = st.sidebar.selectbox("Taxonomy file", file_names, index=default_idx)
src_path = CONV_DIR / selected
kept_path = kept_path_for(src_path)

convs = load_jsonl(src_path)
total = len(convs)
if total == 0:
    st.warning(f"No conversations in {src_path}")
    st.stop()

# Per-file index in session_state
state_key = f"idx::{selected}"
if state_key not in st.session_state:
    st.session_state[state_key] = 0

idx = st.session_state[state_key]
idx = max(0, min(idx, total - 1))
st.session_state[state_key] = idx

kept_ids = load_kept_ids(kept_path)
kept_count = len(kept_ids)

# Sidebar: status + nav
st.sidebar.markdown(f"**Source:** `{src_path.name}`")
st.sidebar.markdown(f"**Kept file:** `{kept_path.name}`")
st.sidebar.markdown(f"**Kept:** {kept_count} / {total}")
st.sidebar.markdown(f"**Position:** {idx + 1} / {total}")

jump = st.sidebar.number_input("Jump to #", min_value=1, max_value=total, value=idx + 1)
if jump - 1 != idx:
    st.session_state[state_key] = jump - 1
    st.rerun()

show_only_unreviewed = st.sidebar.checkbox("Skip already-kept", value=False)


# -----------------------------------------------------------------------------
# Main panel
# -----------------------------------------------------------------------------

conv = convs[idx]
conv_id = conv["id"]
already_kept = conv_id in kept_ids

# Header
header_cols = st.columns([3, 1])
with header_cols[0]:
    st.subheader(f"#{idx + 1}/{total} — {conv.get('scenario', '(no scenario)')}")
    badges = []
    badges.append(f"taxonomy: `{conv.get('taxonomy', '?')}`")
    badges.append(f"difficulty: `{conv.get('difficulty', '?')}`")
    badges.append(f"id: `{conv_id[:8]}`")
    if already_kept:
        badges.append("✅ KEPT")
    st.markdown(" · ".join(badges))
with header_cols[1]:
    if already_kept:
        if st.button("Un-keep", use_container_width=True):
            remove_kept(kept_path, conv_id)
            st.rerun()
    else:
        if st.button("✅ Keep", type="primary", use_container_width=True):
            append_kept(kept_path, conv)
            advance = idx + 1
            if show_only_unreviewed:
                while advance < total and convs[advance]["id"] in (kept_ids | {conv_id}):
                    advance += 1
            if advance >= total:
                advance = total - 1
            st.session_state[state_key] = advance
            st.rerun()

# Turns
st.divider()
st.markdown("### Conversation")
for t in conv["turns"]:
    speaker = t["speaker"].upper()
    text = t.get("text", "")
    img = t.get("image")
    with st.chat_message("user" if t["speaker"] == "user" else "assistant"):
        st.markdown(f"**Turn {t['turn_id']} — {speaker}**")
        st.markdown(text)
        if img:
            sp = img.get("source_path")
            cap = f"{img['id']}: {img['description']}"
            if sp:
                p = IMG_ROOT / sp
                if p.exists():
                    st.image(str(p), caption=cap, use_container_width=True)
                else:
                    st.warning(f"⚠ Image file missing: `{sp}`")
                    st.caption(cap)
            else:
                st.info(f"📷 (image not yet generated) {cap}")

# Question + MCQ + GT
st.divider()
st.markdown("### Final Question")
st.markdown(f"> {conv.get('final_question', '(none)')}")

mcq = conv.get("mcq_options") or []
if mcq:
    st.markdown("**Options:**")
    for opt in mcq:
        marker = "✅" if opt.get("is_correct") else "·"
        st.markdown(f"- {marker} **{opt['label']}.** {opt['text']}")

with st.expander("Reasoning chain & metadata"):
    st.markdown(f"**Ground truth:** {conv.get('ground_truth', '')}")
    st.markdown(f"**Reasoning chain:** {conv.get('reasoning_chain', '')}")
    st.markdown(f"**Why sequential:** {conv.get('why_sequential', '')}")
    st.markdown(f"**Single-turn solvable:** {conv.get('single_turn_solvable')}")
    st.markdown(f"**Num images:** {conv.get('num_images')}")
    if conv.get("metadata"):
        st.json(conv["metadata"])

# Bottom nav
st.divider()
nav = st.columns(4)
with nav[0]:
    if st.button("⏮ First", use_container_width=True):
        st.session_state[state_key] = 0
        st.rerun()
with nav[1]:
    if st.button("◀ Prev", use_container_width=True, disabled=(idx == 0)):
        new_idx = idx - 1
        if show_only_unreviewed:
            while new_idx > 0 and convs[new_idx]["id"] in kept_ids:
                new_idx -= 1
        st.session_state[state_key] = new_idx
        st.rerun()
with nav[2]:
    if st.button("Next ▶", use_container_width=True, disabled=(idx >= total - 1)):
        new_idx = idx + 1
        if show_only_unreviewed:
            while new_idx < total - 1 and convs[new_idx]["id"] in kept_ids:
                new_idx += 1
        st.session_state[state_key] = new_idx
        st.rerun()
with nav[3]:
    if st.button("Last ⏭", use_container_width=True):
        st.session_state[state_key] = total - 1
        st.rerun()
