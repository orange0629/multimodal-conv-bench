#!/usr/bin/env python3
"""
Build a self-contained HTML demo for browsing benchmark conversations.
Images are referenced by local file paths and loaded on demand by the browser.

Usage
-----
  python pipeline/build_demo.py --input-dir pipeline/output_images --output demo.html
  python pipeline/build_demo.py --taxonomy cross_turn_entity_tracking --limit 20
"""

import argparse
import json
import sys
from pathlib import Path


def load_records(input_dir: Path, taxonomy: str | None, limit: int | None) -> list[dict]:
    if input_dir.suffix == ".jsonl":
        paths = [input_dir]
    else:
        conv_paths = sorted(input_dir.rglob("conversation.jsonl"))
        flat_paths = sorted(p for p in input_dir.rglob("*.jsonl") if p.name != "conversation.jsonl")
        paths = conv_paths if conv_paths else flat_paths

    if taxonomy:
        paths = [p for p in paths if taxonomy in str(p)]

    records = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line or '"_error"' in line:
                    continue
                records.append(json.loads(line))

    if limit:
        records = records[:limit]
    return records


def build_records(records: list[dict], html_dir: Path) -> list[dict]:
    out = []
    for i, record in enumerate(records):
        meta = record.get("_meta", {})
        gt = record.get("ground_truth", {})

        turns = []
        for t in record.get("turns", []):
            ip = t.get("image_path")
            # Relative path from the HTML file so the HTTP server can serve it
            img_src = None
            if ip and Path(ip).exists():
                try:
                    img_src = Path(ip).resolve().relative_to(html_dir.resolve()).as_posix()
                except ValueError:
                    img_src = str(Path(ip).resolve())
            turns.append({
                "turn_id": t.get("turn_id"),
                "role": t.get("role"),
                "text": t.get("text", ""),
                "image_description": t.get("image_description", ""),
                "img_src": img_src,
            })

        out.append({
            "idx": i,
            "taxonomy": meta.get("taxonomy", ""),
            "scenario": meta.get("scenario", ""),
            "scenario_title": record.get("scenario_title", ""),
            "scenario_description": record.get("scenario_description", ""),
            "turns": turns,
            "gt_answer": gt.get("answer", ""),
            "gt_question_type": gt.get("question_type", ""),
            "gt_reasoning": gt.get("reasoning_chain", ""),
            "gt_difficulty": gt.get("key_difficulty", ""),
            "gen_model": meta.get("model", ""),
            "generated_at": meta.get("generated_at", ""),
        })
    return out


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Multimodal Conv Benchmark</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f4f5f7;color:#222;display:flex;flex-direction:column;height:100vh;overflow:hidden}

#hdr{background:#16213e;color:#fff;padding:10px 18px;display:flex;align-items:center;gap:12px;flex-shrink:0}
#hdr h1{font-size:15px;font-weight:700;white-space:nowrap}
#hdr select,#hdr input{padding:5px 10px;border:1px solid #3a3a6e;border-radius:6px;background:#1e2a50;color:#fff;font-size:13px}
#hdr input{flex:1;max-width:280px}
#hdr input::placeholder{color:#8899bb}
#cnt{font-size:12px;color:#8899bb;margin-left:auto;white-space:nowrap}

#main{display:flex;flex:1;overflow:hidden}

#sb{width:290px;flex-shrink:0;background:#fff;border-right:1px solid #e2e4ea;overflow-y:auto}
.ci{padding:10px 14px;border-bottom:1px solid #f0f1f4;cursor:pointer;transition:background .1s}
.ci:hover{background:#f0f4ff}
.ci.active{background:#e8ecff;border-left:3px solid #4361ee}
.ci-num{font-size:10px;color:#999;margin-bottom:2px}
.ci-title{font-size:12px;font-weight:500;line-height:1.35;margin-bottom:4px}
.ci-badge{font-size:10px;background:#eef2ff;color:#3a5be0;padding:1px 7px;border-radius:10px;display:inline-block}

#det{flex:1;overflow-y:auto;padding:22px 28px}
.empty{color:#aaa;text-align:center;margin-top:100px;font-size:14px}

.sc-hdr{margin-bottom:18px}
.sc-hdr h2{font-size:17px;font-weight:700;margin-bottom:8px}
.sc-desc{font-size:13px;color:#444;line-height:1.6;background:#fafafa;border-left:3px solid #c8cfe8;padding:10px 14px;border-radius:0 8px 8px 0;margin-bottom:8px}
.sc-scenario{font-size:12px;color:#666;line-height:1.5;background:#f5f5f5;padding:8px 12px;border-radius:6px;margin-bottom:10px}
.chips{display:flex;gap:8px;flex-wrap:wrap}
.chip{font-size:11px;padding:2px 9px;border-radius:10px;background:#f0f0f0;color:#555}
.chip.tax{background:#eef2ff;color:#3a5be0}
.chip.qt{background:#fff3e0;color:#e65100}

.conv{margin:16px 0}
.turn{display:flex;gap:10px;margin-bottom:14px}
.turn.assistant{flex-direction:row-reverse}
.av{width:30px;height:30px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;flex-shrink:0;margin-top:2px}
.turn.user .av{background:#dbeafe;color:#1d4ed8}
.turn.assistant .av{background:#fce7f3;color:#9d174d}
.tc{max-width:72%}
.turn-meta{font-size:10px;color:#bbb;margin-bottom:3px}
.turn.assistant .turn-meta{text-align:right}
.bbl{padding:9px 13px;border-radius:12px;font-size:13px;line-height:1.6;word-break:break-word}
.turn.user .bbl{background:#dbeafe;border-top-left-radius:3px}
.turn.assistant .bbl{background:#fce7f3;border-top-right-radius:3px}
.turn img{max-width:340px;max-height:340px;border-radius:8px;margin-top:8px;border:1px solid #ddd;display:block}
details.desc{margin-top:5px;max-width:340px}
details.desc summary{font-size:11px;color:#bbb;cursor:pointer;list-style:none;user-select:none}
details.desc summary::before{content:"▶ "}
details.desc[open] summary::before{content:"▼ "}
details.desc p{font-size:11px;color:#888;line-height:1.45;margin-top:4px;font-style:italic}

.gt{background:#fff;border:1px solid #e2e4ea;border-radius:10px;padding:14px 18px;margin-top:8px}
.gt-hdr{font-size:13px;font-weight:700;margin-bottom:10px;display:flex;align-items:center;justify-content:space-between;cursor:pointer;user-select:none}
.gt-hdr .tog{font-size:11px;color:#999;font-weight:400}
.gt-grid{display:grid;gap:10px}
.gt-row label{font-size:10px;font-weight:700;color:#999;text-transform:uppercase;letter-spacing:.6px}
.gt-row .v{font-size:13px;line-height:1.5;margin-top:3px}
.ans{display:inline-block;padding:3px 14px;border-radius:20px;font-size:15px;font-weight:700;background:#dcfce7;color:#15803d}

#nav{display:flex;align-items:center;justify-content:center;gap:10px;padding:9px;background:#fff;border-top:1px solid #e2e4ea;flex-shrink:0}
#nav button{padding:5px 18px;border:1px solid #d1d5db;border-radius:6px;background:#fff;cursor:pointer;font-size:13px}
#nav button:hover:not(:disabled){background:#f0f0f0}
#nav button:disabled{opacity:.3;cursor:default}
#nav-pos{font-size:13px;color:#555;min-width:80px;text-align:center}
</style>
</head>
<body>

<div id="hdr">
  <h1>&#127919; Multimodal Conv Benchmark</h1>
  <select id="tax-sel"><option value="">All taxonomies</option></select>
  <input id="search" type="text" placeholder="Search title / scenario...">
  <span id="cnt"></span>
</div>

<div id="main">
  <div id="sb"></div>
  <div id="det"><div class="empty">&#8592; Select a conversation from the sidebar</div></div>
</div>

<div id="nav">
  <button id="btn-p" onclick="go(-1)">&#8592; Prev</button>
  <span id="nav-pos">—</span>
  <button id="btn-n" onclick="go(1)">Next &#8594;</button>
</div>

<script>
const DATA = __DATA__;
let filtered = DATA.slice(), cur = -1;

const taxSel = document.getElementById('tax-sel');
[...new Set(DATA.map(r=>r.taxonomy))].sort().forEach(t=>{
  const o=document.createElement('option'); o.value=t; o.textContent=t.replace(/_/g,' '); taxSel.appendChild(o);
});

function applyFilters(){
  const tax=taxSel.value, q=document.getElementById('search').value.trim().toLowerCase();
  const prevId = cur>=0 ? filtered[cur]?.idx : -1;
  filtered=DATA.filter(r=>(!tax||r.taxonomy===tax)&&(!q||r.scenario_title.toLowerCase().includes(q)||r.scenario.toLowerCase().includes(q)));
  renderSb();
  const ni=filtered.findIndex(r=>r.idx===prevId);
  cur=ni>=0?ni:(filtered.length>0?0:-1);
  cur>=0?renderDet():document.getElementById('det').innerHTML='<div class="empty">No results</div>';
  updateNav();
  document.getElementById('cnt').textContent=`${filtered.length} / ${DATA.length}`;
}
taxSel.addEventListener('change',applyFilters);
document.getElementById('search').addEventListener('input',applyFilters);

function renderSb(){
  document.getElementById('sb').innerHTML=filtered.map((r,i)=>`
    <div class="ci ${i===cur?'active':''}" onclick="sel(${i})">
      <div class="ci-num">#${r.idx+1}</div>
      <div class="ci-title">${esc(r.scenario_title||r.scenario.slice(0,70))}</div>
      <span class="ci-badge">${r.taxonomy.replace(/_/g,' ')}</span>
    </div>`).join('');
}

function sel(i){cur=i;renderSb();renderDet();updateNav();document.querySelectorAll('.ci')[i]?.scrollIntoView({block:'nearest'});}
function go(d){const n=cur+d;if(n>=0&&n<filtered.length)sel(n);}
function updateNav(){
  document.getElementById('btn-p').disabled=cur<=0;
  document.getElementById('btn-n').disabled=cur<0||cur>=filtered.length-1;
  document.getElementById('nav-pos').textContent=cur>=0?`${cur+1} / ${filtered.length}`:'—';
}
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;if(e.key==='ArrowLeft')go(-1);if(e.key==='ArrowRight')go(1);});

function renderDet(){
  const r=filtered[cur]; if(!r)return;
  const turns=r.turns.map(t=>`
    <div class="turn ${t.role}">
      <div class="av">${t.role==='user'?'U':'A'}</div>
      <div class="tc">
        <div class="turn-meta">Turn ${t.turn_id} &middot; ${t.role}</div>
        <div class="bbl">${esc(t.text)}</div>
        ${t.img_src?`<img src="${t.img_src}" alt="turn ${t.turn_id}" onerror="this.style.display='none'">`:''}
        ${t.image_description?`<details class="desc"><summary>image description</summary><p>${esc(t.image_description)}</p></details>`:''}
      </div>
    </div>`).join('');

  document.getElementById('det').innerHTML=`
    <div class="sc-hdr">
      <h2>${esc(r.scenario_title||'Conversation #'+(r.idx+1))}</h2>
      ${r.scenario_description?`<div class="sc-desc">${esc(r.scenario_description)}</div>`:''}
      ${r.scenario?`<div class="sc-scenario">${esc(r.scenario)}</div>`:''}
      <div class="chips">
        <span class="chip tax">${r.taxonomy.replace(/_/g,' ')}</span>
        <span class="chip qt">${r.gt_question_type}</span>
        ${r.gen_model?`<span class="chip">&#129302; ${r.gen_model.split('/').pop()}</span>`:''}
        ${r.generated_at?`<span class="chip">${r.generated_at.slice(0,10)}</span>`:''}
      </div>
    </div>
    <div class="conv">${turns}</div>
    <div class="gt">
      <div class="gt-hdr" onclick="toggleGt()">&#127919; Ground Truth<span class="tog" id="gt-tog">[click to hide]</span></div>
      <div class="gt-grid" id="gt-body">
        <div class="gt-row"><label>Answer</label><div class="v"><span class="ans">${esc(r.gt_answer)}</span></div></div>
        <div class="gt-row"><label>Question type</label><div class="v">${esc(r.gt_question_type)}</div></div>
        <div class="gt-row"><label>Reasoning chain</label><div class="v">${esc(r.gt_reasoning)}</div></div>
        <div class="gt-row"><label>Key difficulty</label><div class="v">${esc(r.gt_difficulty)}</div></div>
      </div>
    </div>`;
}

function toggleGt(){
  const b=document.getElementById('gt-body'),t=document.getElementById('gt-tog');
  const h=b.style.display==='none'; b.style.display=h?'':'none'; t.textContent=h?'[click to hide]':'[click to show]';
}

function esc(s){return(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>');}

applyFilters();
</script>
</body>
</html>
"""


def main():
    p = argparse.ArgumentParser(
        description="Build an HTML demo for browsing benchmark conversations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input-dir", default="pipeline/output_images")
    p.add_argument("--output", default="demo.html")
    p.add_argument("--taxonomy", default=None)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    print(f"[demo] loading from {input_dir} ...", file=sys.stderr)
    records = load_records(input_dir, args.taxonomy, args.limit)
    print(f"[demo] {len(records)} conversations", file=sys.stderr)

    out = Path(args.output).resolve()
    demo_data = build_records(records, html_dir=out.parent)
    html = HTML.replace("__DATA__", json.dumps(demo_data, ensure_ascii=False))
    out.write_text(html, encoding="utf-8")
    size_kb = out.stat().st_size / 1024
    print(f"[demo] written → {out}  ({size_kb:.0f} KB)", file=sys.stderr)


if __name__ == "__main__":
    main()
