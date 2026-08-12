"""Renderer for RIZZMATICS // THE UNMODELED — the private narrative.

A single guided journey (not a documentation sidebar): a quiet chapter rail, one
card at a time, continue / back, and a visual arc that runs software → memories
→ human as you go deeper. Premium minimal: true black, one neon-pink accent,
white type, hairlines, a faint starfield, an elegant serif at the emotional
beats. No gradients, no dating-app chrome, no hearts everywhere.

Security is unchanged and still server-side: the rich component (and the node
data inside it) is only ever produced inside the authenticated branch. Before
auth, only the door + hint are rendered — no lore. See
``tests/test_after_hours_security.py``.
"""

from __future__ import annotations

import json

from streamlit.components.v1 import html as components_html

from . import gate
from .lorekit import LoreRegistry, Node

_STATE_KEY = "_unmodeled_state"
_FONTS = ("https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700"
          "&family=Fraunces:ital,opsz,wght@1,9..144,400;1,9..144,500;0,9..144,500"
          "&family=JetBrains+Mono:wght@400;500;600&display=swap")

_SIGIL = ('<svg class="sig" width="13" height="13" viewBox="0 0 13 13" fill="none">'
          '<rect x="6.5" y="0.7" width="8.2" height="8.2" rx="1.4" transform="rotate(45 6.5 0.7)" '
          'stroke="#ff2e88" stroke-width="1.1"/></svg>')


# --------------------------------------------------------------------------- #
# Chrome + door CSS (injected around the Streamlit-side door)
# --------------------------------------------------------------------------- #
def _chrome_and_door_css() -> str:
    return f"""
    <style>
      @import url('{_FONTS}');
      #MainMenu, header[data-testid="stHeader"], footer {{ display:none !important; }}
      [data-testid="stToolbar"], [data-testid="stDecoration"] {{ display:none !important; }}
      .block-container {{ padding:0 !important; max-width:100% !important; }}
      .stApp {{ background:#000; }}

      .um-door {{
        font-family:'JetBrains Mono',monospace; color:#f2f2f4;
        max-width:560px; margin:12vh auto 0; padding:46px 46px 22px;
        background:#050505; border:1px solid rgba(255,255,255,0.09); border-radius:14px; }}
      .um-door .mark {{ font-family:'Space Grotesk',sans-serif; font-weight:700; letter-spacing:.26em;
        font-size:1.35rem; color:#fff; display:flex; align-items:center; gap:12px; }}
      .um-door .name {{ font-family:'Space Grotesk',sans-serif; font-weight:500; letter-spacing:.42em;
        font-size:.72rem; color:#ff2e88; text-transform:uppercase; margin-top:14px; }}
      .um-door .rule {{ height:1px; background:rgba(255,255,255,0.09); margin:24px 0; }}
      .um-door .lede {{ white-space:pre-wrap; line-height:2.0; font-size:.86rem; color:rgba(242,242,244,0.7); }}
      .um-door .hint {{ margin:24px 0 6px; font-size:.78rem; color:rgba(242,242,244,0.42);
        border:1px solid rgba(255,255,255,0.08); border-left:2px solid #ff2e88; border-radius:6px;
        padding:12px 14px; letter-spacing:.05em; }}
      .um-door .acc {{ font-size:.6rem; letter-spacing:.36em; color:rgba(242,242,244,0.5);
        text-transform:uppercase; margin:8px 2px 0; }}

      .stForm {{ border:none !important; background:transparent !important; padding:0 !important; }}
      [data-testid="stTextInput"] input {{
        background:#000 !important; color:#fff !important; border:1px solid rgba(255,255,255,0.14) !important;
        border-radius:9px !important; font-family:'JetBrains Mono',monospace !important;
        letter-spacing:.22em; padding:12px 14px !important; }}
      [data-testid="stTextInput"] input:focus {{
        border-color:#ff2e88 !important; box-shadow:0 0 0 1px #ff2e88 !important; }}
      [data-testid="stFormSubmitButton"] button {{
        font-family:'Space Grotesk',sans-serif !important; font-weight:600 !important; letter-spacing:.3em !important;
        color:#ff2e88 !important; background:#000 !important; border:1px solid #ff2e88 !important;
        border-radius:9px !important; width:100%; padding:9px 0 !important; transition:background .15s,color .15s; }}
      [data-testid="stFormSubmitButton"] button:hover {{ background:#ff2e88 !important; color:#000 !important; }}
    </style>
    """


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def render_after_hours(st) -> None:
    st.markdown(_chrome_and_door_css(), unsafe_allow_html=True)
    registry = gate.load_registry()
    if registry is None:
        _render_no_module(st)
        return
    if not gate.password_configured():
        _render_unconfigured(st)
        return
    if not gate.is_authenticated(st.session_state):
        _render_door(st, registry)
        return
    _render_experience(st, registry)


# --------------------------------------------------------------------------- #
# Inert states
# --------------------------------------------------------------------------- #
def _render_no_module(st) -> None:
    st.markdown(
        f'<div class="um-door"><div class="mark">{_SIGIL}RIZZMATICS</div>'
        '<div class="name">The Unmodeled</div><div class="rule"></div>'
        '<div class="lede">This build does not include the private module.</div></div>',
        unsafe_allow_html=True)


def _render_unconfigured(st) -> None:
    st.markdown(
        f'<div class="um-door"><div class="mark">{_SIGIL}RIZZMATICS</div>'
        '<div class="name">The Unmodeled</div><div class="rule"></div>'
        '<div class="lede">Private module present, but no access key is configured.\n'
        'Run: python scripts/set_private_password.py</div></div>',
        unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# The door (pre-auth: door + hint only, never lore)
# --------------------------------------------------------------------------- #
def _render_door(st, registry: LoreRegistry) -> None:
    st.markdown(
        f"""
        <div class="um-door">
          <div class="mark">{_SIGIL}RIZZMATICS</div>
          <div class="name">The Unmodeled</div>
          <div class="rule"></div>
          <div class="lede">Some things can be measured.
Some things can be remembered.
Some things are better left to the people who were there.</div>
          <div class="hint">{registry.clue}</div>
          <div class="acc">access</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns([1, 4, 1])
    with cols[1]:
        with st.form("unmodeled_auth", clear_on_submit=False):
            pw = st.text_input("Access", type="password",
                               label_visibility="collapsed", placeholder="")
            submitted = st.form_submit_button("ENTER")
    if submitted:
        if gate.check_password(pw):
            gate.mark_authenticated(st.session_state)
            st.rerun()
        else:
            st.error("Not quite.")


# --------------------------------------------------------------------------- #
# Authenticated experience — the bespoke journey component
# --------------------------------------------------------------------------- #
def _nodes_payload(registry: LoreRegistry) -> list[dict]:
    return [
        {"id": n.id, "title": n.title, "kind": n.kind, "status": n.status,
         "body": list(n.body)}
        for n in registry.all()
    ]


def authenticated_component_html(registry: LoreRegistry) -> str:
    """Return the full self-contained journey as one HTML string (post-auth only)."""
    payload = json.dumps(_nodes_payload(registry))
    return (_SPA_TEMPLATE
            .replace("/*__NODES__*/", payload)
            .replace("__FONTS__", _FONTS)
            .replace("__SIGIL__", _SIGIL))


def _render_experience(st, registry: LoreRegistry) -> None:
    components_html(authenticated_component_html(registry), height=940, scrolling=True)


# --------------------------------------------------------------------------- #
# The journey component (HTML + CSS + JS, no external deps)
# --------------------------------------------------------------------------- #
_SPA_TEMPLATE = r"""
<!doctype html><html><head><meta charset="utf-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="__FONTS__" rel="stylesheet">
<style>
  :root{ --bg:#000; --panel:#050505; --line:rgba(255,255,255,0.09);
    --white:#fff; --ink:#eceef2; --mut:rgba(236,238,242,0.42); --mut2:rgba(236,238,242,0.62); --pink:#ff2e88; }
  *{box-sizing:border-box;}
  html,body{margin:0;padding:0;background:var(--bg);}
  ::selection{background:var(--pink); color:#000;}
  #um{ font-family:'JetBrains Mono',monospace; color:var(--ink); background:var(--bg);
    min-height:940px; position:relative; overflow:hidden; }
  #stars{position:absolute; inset:0; z-index:0; pointer-events:none;}
  #stars .s{position:absolute; border-radius:50%; background:#fff;}
  @keyframes tw{0%,100%{opacity:var(--o);}50%{opacity:calc(var(--o)*0.25);}}
  .wrap{position:relative; z-index:1; max-width:1080px; margin:0 auto; padding:36px 40px 64px;}

  .top{display:flex; justify-content:space-between; align-items:center; gap:24px;
    padding-bottom:20px; border-bottom:1px solid var(--line); margin-bottom:30px;}
  .mark{font-family:'Space Grotesk',sans-serif; font-weight:700; letter-spacing:.26em;
    font-size:1.1rem; color:var(--white); display:flex; align-items:center; gap:12px;}
  .mark .nm{font-family:'JetBrains Mono',monospace; font-weight:400; font-size:.56rem;
    letter-spacing:.4em; color:var(--pink); text-transform:uppercase; margin-left:4px; align-self:flex-end; margin-bottom:2px;}
  .pbar{flex:1; max-width:320px; height:2px; background:rgba(255,255,255,0.12); position:relative;}
  .pbar i{position:absolute; left:0; top:0; bottom:0; background:var(--pink); transition:width .35s ease;}
  .pnum{font-size:.62rem; letter-spacing:.2em; color:var(--mut2); min-width:66px; text-align:right;}

  .grid{display:grid; grid-template-columns:210px 1fr; gap:48px;}
  @media(max-width:800px){.grid{grid-template-columns:1fr; gap:26px;}}
  .toc{display:flex; flex-direction:column; gap:2px;}
  .toc .cap{font-size:.56rem; letter-spacing:.34em; color:var(--mut); text-transform:uppercase; margin-bottom:14px;}
  .tocitem{display:block; width:100%; font-size:.74rem; letter-spacing:.02em; color:var(--mut);
    background:none; border:none; text-align:left; padding:7px 0 7px 14px; cursor:pointer;
    position:relative; transition:color .14s;}
  .tocitem .n{color:rgba(236,238,242,0.28); margin-right:9px; font-size:.66rem;}
  .tocitem:hover{color:var(--ink);}
  .tocitem.seen{color:var(--mut2);}
  .tocitem.cur{color:var(--white);}
  .tocitem.cur:before{content:""; position:absolute; left:0; top:50%; transform:translateY(-50%);
    width:2px; height:15px; background:var(--pink);}

  .stage{min-height:560px; display:flex; flex-direction:column;}
  .card{background:var(--panel); border:1px solid var(--line); border-radius:12px;
    padding:44px 48px; animation:fade .3s ease; flex:1;}
  @keyframes fade{from{opacity:0; transform:translateY(8px);}to{opacity:1; transform:none;}}
  .kick{font-size:.56rem; letter-spacing:.34em; text-transform:uppercase; color:var(--pink);}
  .card h1{font-family:'Space Grotesk',sans-serif; font-weight:600; letter-spacing:.02em;
    font-size:1.55rem; color:var(--white); margin:14px 0 22px;}
  .body{white-space:pre-wrap; line-height:1.85; font-size:.92rem; color:var(--ink);}

  /* the arc: softer + serif as it gets more human */
  .card.landing{border-color:var(--line);}
  .card.landing h1{font-family:'Space Grotesk',sans-serif; letter-spacing:.16em; font-size:1.9rem;}
  .card.landing .body{color:var(--mut2); line-height:2.0;}
  .card.landing .restored{font-size:.6rem; letter-spacing:.34em; text-transform:uppercase; color:var(--pink); margin-bottom:6px;}

  .card.quiet{background:transparent; border-color:rgba(255,255,255,0.05);}
  .card.quiet .body{color:var(--mut2); line-height:2.1; font-size:.98rem;}

  .card.override h1{color:var(--pink);}

  .card.peak{background:transparent; border:none; padding:56px 40px; text-align:center;}
  .card.peak .heart{color:var(--pink); font-size:1rem; letter-spacing:.3em; margin-bottom:18px;}
  .card.peak h1{font-family:'Fraunces',serif; font-style:italic; font-weight:500; letter-spacing:0;
    font-size:1.8rem; color:var(--white); margin-bottom:34px;}
  .card.peak .body{font-family:'Fraunces',serif; font-style:italic; color:#f5f5f7;
    font-size:1.24rem; line-height:2.1; max-width:620px; margin:0 auto;}

  .card.final{background:transparent; border:none; text-align:center; padding:52px 40px;}
  .card.final h1{font-family:'Fraunces',serif; font-style:italic; font-weight:500; font-size:1.7rem; letter-spacing:0;}
  .card.final .body{font-family:'Fraunces',serif; font-style:italic; font-size:1.16rem; line-height:2.0;
    color:#f2f2f4; max-width:600px; margin:0 auto;}
  .mmode{font-family:'Space Grotesk',sans-serif; font-weight:600; letter-spacing:.3em; color:var(--pink);
    font-size:.86rem; margin:30px 0 0;}
  .brand{margin-top:44px; padding-top:30px; border-top:1px solid var(--line); max-width:520px; margin-left:auto; margin-right:auto;}
  .brand .bw{font-family:'Space Grotesk',sans-serif; font-weight:700; letter-spacing:.3em; color:#fff; font-size:1.1rem;}
  .brand .tl{font-size:.74rem; color:var(--mut2); margin-top:12px; letter-spacing:.02em;}
  .brand .bc{font-size:.72rem; color:var(--mut); margin-top:16px; line-height:1.8; white-space:pre-wrap; font-style:normal;}

  /* dice / game motif */
  .dice{display:flex; gap:14px; margin:6px 0 26px;}
  .die{width:46px; height:46px; border:1px solid var(--line); border-radius:9px; background:#0a0a0c;
    display:grid; grid-template-columns:repeat(3,1fr); grid-template-rows:repeat(3,1fr); padding:8px; gap:2px;}
  .die span{width:6px; height:6px; border-radius:50%; align-self:center; justify-self:center;}
  .die span.on{background:var(--pink); box-shadow:0 0 6px rgba(255,46,136,0.5);}

  /* controls */
  .controls{display:flex; justify-content:space-between; align-items:center; margin-top:22px;}
  .btn{font-family:'Space Grotesk',sans-serif; font-weight:600; letter-spacing:.14em; font-size:.72rem;
    background:none; border:1px solid var(--line); color:var(--ink); padding:11px 20px; border-radius:9px;
    cursor:pointer; transition:all .15s;}
  .btn.next{border-color:var(--pink); color:var(--pink);}
  .btn.next:hover{background:var(--pink); color:#000;}
  .btn.back:hover{color:var(--white); border-color:rgba(255,255,255,0.25);}
  .btn.ghost{visibility:hidden;}
  .endnote{font-size:.6rem; letter-spacing:.3em; text-transform:uppercase; color:var(--mut);}
</style></head>
<body>
<div id="um">
  <div id="stars"></div>
  <div class="wrap">
    <div class="top">
      <div class="mark">__SIGIL__RIZZMATICS <span class="nm">the unmodeled</span></div>
      <div class="pbar"><i id="pbar"></i></div>
      <div class="pnum" id="pnum"></div>
    </div>
    <div class="grid">
      <div class="toc"><div class="cap">contents</div><div id="toc"></div></div>
      <div class="stage"><div id="card"></div>
        <div class="controls" id="controls"></div>
      </div>
    </div>
  </div>
</div>
<script>
  (function(){ const box=document.getElementById('stars'), N=44;
    for(let i=0;i<N;i++){ const s=document.createElement('div'); s.className='s';
      const sz=Math.random()<0.82?1:2, o=0.07+Math.random()*0.32;
      s.style.width=s.style.height=sz+'px'; s.style.left=(Math.random()*100)+'%'; s.style.top=(Math.random()*100)+'%';
      s.style.setProperty('--o',o); s.style.opacity=o;
      if(Math.random()<0.15) s.style.background='#ff2e88';
      if(Math.random()<0.4){ s.style.animation='tw '+(4+Math.random()*6)+'s ease-in-out infinite'; s.style.animationDelay=(Math.random()*6)+'s'; }
      box.appendChild(s);} })();

  const NODES = /*__NODES__*/;
  const esc = s => String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
  const pad = n => String(n).padStart(2,'0');
  const CONTENT = NODES.length - 1;           // pages after the landing
  let i = 0; const seen = new Set([0]);

  function die(pips){ // pips: array of 9 booleans (grid positions)
    return '<div class="die">'+pips.map(p=>`<span class="${p?'on':''}"></span>`).join('')+'</div>';
  }
  const SIX = [1,0,1, 1,0,1, 1,0,1].map(Boolean);

  function cardHTML(n, idx){
    if(n.kind==='landing'){
      return `<div class="card landing">
        <div class="restored">${esc(n.status||'context restored')}</div>
        <h1>${esc(n.title)}</h1>
        <div class="body">${esc(n.body.join('\n'))}</div></div>`;
    }
    if(n.kind==='peak'){
      return `<div class="card peak"><div class="heart">&#10084;</div>
        <h1>${esc(n.title)}</h1><div class="body">${esc(n.body.join('\n'))}</div></div>`;
    }
    if(n.kind==='final'){
      return `<div class="card final">
        <h1>${esc(n.title)}</h1><div class="body">${esc(n.body.join('\n'))}</div>
        <div class="mmode">MADISON MODE: ON.</div>
        <div class="brand"><div class="bw">RIZZMATICS</div>
          <div class="tl">Applied mathematics for completely unnecessary flirting.</div>
          <div class="bc">Built because apparently\n"let's just see what happens in Madison"\nwasn't sufficiently computational.</div>
        </div></div>`;
    }
    const dice = n.kind==='game' ? `<div class="dice">${die(SIX)}${die(SIX)}</div>` : '';
    return `<div class="card ${esc(n.kind)}">
      <div class="kick">${pad(idx)} &nbsp;·&nbsp; the unmodeled</div>
      <h1>${esc(n.title)}</h1>${dice}
      <div class="body">${esc(n.body.join('\n'))}</div></div>`;
  }

  function renderTOC(){
    const t = document.getElementById('toc'); t.innerHTML='';
    NODES.forEach((n,idx)=>{
      const el = document.createElement('button');
      el.className='tocitem'+(idx===i?' cur':'')+(seen.has(idx)&&idx!==i?' seen':'');
      const num = idx===0 ? '' : `<span class="n">${pad(idx)}</span>`;
      el.innerHTML = num + esc(n.title);
      el.onclick = ()=>go(idx);
      t.appendChild(el);
    });
  }
  function renderControls(){
    const c = document.getElementById('controls'); const n = NODES[i];
    const back = i>0 ? `<button class="btn back" onclick="go(${i-1})">&larr; back</button>`
                     : `<button class="btn ghost">back</button>`;
    let right;
    if(i===0) right = `<button class="btn next" onclick="go(1)">begin &rarr;</button>`;
    else if(i < NODES.length-1) right = `<button class="btn next" onclick="go(${i+1})">continue &rarr;</button>`;
    else right = `<span class="endnote">— the end —</span>`;
    c.innerHTML = back + right;
  }
  function renderProgress(){
    document.getElementById('pbar').style.width = (i/(NODES.length-1)*100)+'%';
    document.getElementById('pnum').textContent = i===0 ? 'the unmodeled' : (pad(i)+' / '+pad(CONTENT));
  }
  function render(){
    const n = NODES[i];
    document.getElementById('card').innerHTML = cardHTML(n, i);
    renderTOC(); renderControls(); renderProgress();
    document.querySelector('.wrap').scrollIntoView({block:'start'});
  }
  function go(idx){ i = Math.max(0, Math.min(NODES.length-1, idx)); seen.add(i); render(); }
  window.go = go;
  render();
</script>
</body></html>
"""
