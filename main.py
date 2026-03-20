import streamlit as st
import replicate
import base64
import cv2
import numpy as np
import io
from PIL import Image

st.set_page_config(layout="wide", page_title="Age Transformer", page_icon="🪞")

# ── CV helpers ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

face_cascade = load_face_cascade()

def draw_base_frame(img_bgr, faces, scan_x_offset=None):
    out   = img_bgr.copy()
    NEON  = (0, 225, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    for (x, y, w, h) in faces:
        overlay = out.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), NEON, -1)
        cv2.addWeighted(overlay, 0.08, out, 0.92, 0, out)
        if scan_x_offset is not None:
            bx    = x + scan_x_offset
            bar_w = max(6, w // 10)
            bx    = max(x, min(bx, x + w - bar_w))
            bar_overlay = out.copy()
            cv2.rectangle(bar_overlay, (bx, y+2), (bx+bar_w, y+h-2), NEON, -1)
            cv2.addWeighted(bar_overlay, 0.55, out, 0.45, 0, out)
            cv2.line(out, (bx+bar_w, y+2), (bx+bar_w, y+h-2), WHITE, 2, cv2.LINE_AA)
        def dashed_rect(img, x, y, w, h, color, thickness=2, dash=16, gap=8):
            for (x1,y1),(x2,y2) in [
                ((x,y),(x+w,y)),((x+w,y),(x+w,y+h)),
                ((x+w,y+h),(x,y+h)),((x,y+h),(x,y))
            ]:
                dist = int(np.hypot(x2-x1, y2-y1))
                if dist == 0: continue
                dx, dy = (x2-x1)/dist, (y2-y1)/dist
                pos, on = 0, True
                while pos < dist:
                    seg = min(pos+(dash if on else gap), dist)
                    if on:
                        cv2.line(img,
                                 (int(x1+dx*pos), int(y1+dy*pos)),
                                 (int(x1+dx*seg), int(y1+dy*seg)),
                                 color, thickness, cv2.LINE_AA)
                    pos += dash if on else gap
                    on = not on
        dashed_rect(out, x, y, w, h, NEON, thickness=2)
        arm = max(20, w // 7)
        for (cx, cy), (sx, sy) in [
            ((x,   y  ),( 1, 1)), ((x+w, y  ),(-1, 1)),
            ((x,   y+h),( 1,-1)), ((x+w, y+h),(-1,-1)),
        ]:
            cv2.line(out, (cx, cy), (cx+sx*arm, cy), WHITE, 3, cv2.LINE_AA)
            cv2.line(out, (cx, cy), (cx, cy+sy*arm), WHITE, 3, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), 5, NEON, -1, cv2.LINE_AA)
        banner_h = 28
        by = max(0, y - banner_h)
        cv2.rectangle(out, (x, by), (x+w, y), NEON, -1)
        label = "ANALIZZO..." if scan_x_offset is not None else "Volto rilevato"
        font, fscale, fthick = cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1
        (tw, th), _ = cv2.getTextSize(label, font, fscale, fthick)
        cv2.putText(out, label,
                    (x+(w-tw)//2, by+(banner_h+th)//2),
                    font, fscale, BLACK, fthick, cv2.LINE_AA)
    return out

def create_scanning_gif(pil_image, faces):
    img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    max_dim = 640
    h0, w0  = img_bgr.shape[:2]
    scale   = min(1.0, max_dim / max(h0, w0))
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w0*scale), int(h0*scale)))
        faces   = [(int(x*scale), int(y*scale), int(w*scale), int(h*scale)) for (x,y,w,h) in faces]
    N_FRAMES = 15; LOOP_REPS = 4; FRAME_DUR = 50
    pil_frames = []
    for _ in range(LOOP_REPS):
        for i in range(N_FRAMES):
            t = i / (N_FRAMES - 1)
            for (x, y, w, h) in faces: offset = int(t * w)
            pil_frames.append(Image.fromarray(cv2.cvtColor(
                draw_base_frame(img_bgr, faces, offset), cv2.COLOR_BGR2RGB)))
        for i in range(N_FRAMES):
            t = 1.0 - i / (N_FRAMES - 1)
            for (x, y, w, h) in faces: offset = int(t * w)
            pil_frames.append(Image.fromarray(cv2.cvtColor(
                draw_base_frame(img_bgr, faces, offset), cv2.COLOR_BGR2RGB)))
    buf = io.BytesIO()
    pil_frames[0].save(buf, format="GIF", save_all=True, append_images=pil_frames[1:],
                       loop=0, duration=FRAME_DUR, optimize=False)
    return buf.getvalue()

def detect_and_annotate(pil_image):
    """Returns (annotated_pil, face_found, faces, gif_bytes_or_None)."""
    img_bgr  = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray     = cv2.equalizeHist(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
    all_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                              minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)
    faces = []
    if len(all_faces) > 0:
        faces = [max(all_faces, key=lambda f: f[2]*f[3])]
    annotated = draw_base_frame(img_bgr, faces, scan_x_offset=None)
    annotated_pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    return annotated_pil, len(faces) > 0, faces


# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #0a0e1a;
    color: #e0e8ff;
}
.stApp {
    background: radial-gradient(ellipse at 20% 0%, #0d1b3e 0%, #0a0e1a 60%);
    min-height: 100vh;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px; }

.phase-stepper {
    display: flex; align-items: center; justify-content: center;
    gap: 0; margin: 0 auto 2.5rem auto; max-width: 640px;
    font-family: 'Share Tech Mono', monospace;
}
.phase-step { display: flex; flex-direction: column; align-items: center; gap: 6px; flex: 1; }
.phase-dot {
    width: 36px; height: 36px; border-radius: 50%;
    border: 2px solid #2a3a6a; background: #10182e;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 700; color: #3a5a9a;
    position: relative; z-index: 2;
}
.phase-dot.active {
    border-color: #00e5ff;
    background: linear-gradient(135deg, #001f3f, #003366);
    color: #00e5ff; box-shadow: 0 0 16px #00e5ff66, 0 0 4px #00e5ff;
}
.phase-dot.done {
    border-color: #00ff9d;
    background: linear-gradient(135deg, #001a10, #003322);
    color: #00ff9d; box-shadow: 0 0 10px #00ff9d44;
}
.phase-label { font-size: 10px; letter-spacing: 2px; text-transform: uppercase; color: #3a5a9a; }
.phase-label.active { color: #00e5ff; }
.phase-label.done   { color: #00ff9d; }
.phase-connector { flex: 1; height: 2px; margin-bottom: 22px; background: #2a3a6a; }
.phase-connector.done { background: linear-gradient(90deg, #00ff9d, #00e5ff); }

.phase-card {
    background: linear-gradient(145deg, #111a32, #0d1424);
    border: 1px solid #1e2e50; border-radius: 16px;
    padding: 2rem 2.5rem; margin-bottom: 1.5rem;
    box-shadow: 0 4px 40px #00000066;
}
.phase-card h2 {
    font-family: 'Share Tech Mono', monospace; color: #00e5ff;
    font-size: 1rem; letter-spacing: 3px; text-transform: uppercase;
    margin-bottom: 1.5rem; border-bottom: 1px solid #1e2e50; padding-bottom: .75rem;
}

.stButton > button {
    background: linear-gradient(135deg, #003366, #001f3f);
    border: 1px solid #00e5ff; color: #00e5ff;
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 2px; font-size: 0.85rem;
    padding: .65rem 2rem; border-radius: 6px; transition: all .3s;
}
.stButton > button:hover { background: #00e5ff; color: #0a0e1a; box-shadow: 0 0 20px #00e5ff66; }
.stButton > button:disabled { border-color: #2a3a6a !important; color: #2a3a6a !important; background: #0a0e1a !important; }

[data-testid="stCameraInput"] > div {
    border: 1px solid #1e2e50 !important; border-radius: 12px !important;
    overflow: hidden; background: #0a0e1a !important;
}
.stAlert { border-radius: 8px; font-family: 'Share Tech Mono', monospace; font-size: .85rem; letter-spacing: 1px; }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
if "phase" not in st.session_state:
    st.session_state["phase"] = "foto"


# ── UI helpers ─────────────────────────────────────────────────────────────────
def render_stepper(phase: str):
    steps = [("01","FOTO","foto"), ("02","ANALISI","analisi"), ("03","RISULTATO","risultato")]
    order = ["foto","analisi","risultato"]
    cur   = order.index(phase)
    html  = '<div class="phase-stepper">'
    for i, (num, lbl, key) in enumerate(steps):
        idx = order.index(key)
        if idx < cur:
            dot_cls, lbl_cls, icon = "phase-dot done", "phase-label done", "✓"
        elif idx == cur:
            dot_cls, lbl_cls, icon = "phase-dot active", "phase-label active", num
        else:
            dot_cls, lbl_cls, icon = "phase-dot", "phase-label", num
        html += (f'<div class="phase-step"><div class="{dot_cls}">{icon}</div>'
                 f'<span class="{lbl_cls}">{lbl}</span></div>')
        if i < len(steps) - 1:
            conn = "phase-connector done" if cur > idx else "phase-connector"
            html += f'<div class="{conn}"></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def card(title):
    st.markdown(f'<div class="phase-card"><h2>{title}</h2>', unsafe_allow_html=True)

def card_end():
    st.markdown('</div>', unsafe_allow_html=True)

def mono(text, color="#3a6a9a"):
    st.markdown(
        f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:.7rem;'
        f'color:{color};letter-spacing:2px;margin-bottom:.5rem;">{text}</div>',
        unsafe_allow_html=True)


# ── Page title ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-bottom:2rem;">
  <div style="font-family:'Rajdhani',sans-serif;font-weight:700;font-size:2rem;color:#e0e8ff;letter-spacing:4px;">
    AGE TRANSFORMER
  </div>
  <div style="font-family:'Share Tech Mono',monospace;font-size:.7rem;letter-spacing:4px;color:#3a6a9a;margin-top:.3rem;">
    TRASFORMAZIONE BIOMETRICA DEL VOLTO
  </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📖 Anteprima", "🪞 Applicazione"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        try:    st.image("start_image.jpg", use_container_width=True)
        except: st.info("start_image.jpg non trovata")
    with col2:
        try:    st.image("transformation.gif", use_container_width=True)
        except: st.info("transformation.gif non trovata")

with tab2:
    phase = st.session_state["phase"]
    render_stepper(phase)

    # ══════════════════════════════════════════════════════════════
    #  PHASE 1 — FOTO
    # ══════════════════════════════════════════════════════════════
    if phase == "foto":
        card("ACQUISISCI IMMAGINE")
        _, cam_col, _ = st.columns([1, 2, 1])
        with cam_col:
            camera_photo = st.camera_input("Scatta un'immagine", label_visibility="collapsed")
            st.markdown("""
            <div style="text-align:center;margin-top:.75rem;
                        font-family:'Share Tech Mono',monospace;font-size:.7rem;
                        color:#3a6a9a;letter-spacing:2px;">
                POSIZIONA IL VOLTO AL CENTRO • BUONA ILLUMINAZIONE
            </div>""", unsafe_allow_html=True)
        card_end()

        card("PARAMETRI TRASFORMAZIONE")
        target_age = st.slider("Età target", min_value=10, max_value=100, value=70, step=5)
        st.markdown(
            f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:.72rem;'
            f'color:#3a6a9a;letter-spacing:2px;margin-top:-.5rem;">'
            f'TARGET AGE: <span style="color:#00e5ff;">{target_age} ANNI</span></div>',
            unsafe_allow_html=True)
        card_end()

        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            if camera_photo:
                st.success("✔  Foto acquisita — pronto per l'analisi")
                if st.button("▶  AVVIA ANALISI", use_container_width=True):
                    st.session_state["image_data"] = camera_photo.getvalue()
                    st.session_state["target_age"] = target_age
                    st.session_state["phase"]      = "analisi"
                    st.rerun()
            else:
                st.warning("⚠  Scatta una foto per continuare")

    # ══════════════════════════════════════════════════════════════
    #  PHASE 2 — ANALISI
    # ══════════════════════════════════════════════════════════════
    elif phase == "analisi":
        image_data = st.session_state["image_data"]
        target_age = st.session_state["target_age"]
        raw_pil    = Image.open(io.BytesIO(image_data)).convert("RGB")

        # 1. Build scanning GIF synchronously (fast, ~1s)
        annotated_pil, face_found, faces = detect_and_annotate(raw_pil)
        if face_found:
            gif_bytes = create_scanning_gif(raw_pil, faces)
        else:
            gif_bytes = None

        # 2. Show GIF while API runs
        card("ANALISI IN CORSO")
        _, gif_col, _ = st.columns([1, 2, 1])
        with gif_col:
            mono("SCANSIONE FACCIALE")
            if gif_bytes:
                st.image(gif_bytes, use_container_width=True)
            else:
                st.image(annotated_pil, use_container_width=True)
        card_end()

        # 3. Call Replicate API
        b64      = base64.b64encode(image_data).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{b64}"
        try:
            output = replicate.run(
                "yuval-alaluf/sam:9222a21c181b707209ef12b5e0d7e94c994b58f01c7b2fec075d2e892362f13c",
                input={"image": data_uri, "target_age": str(target_age)},
            )
            st.session_state["result_bytes"] = output.read()
            st.session_state.pop("result_error", None)
        except Exception as e:
            st.session_state["result_error"] = str(e)

        st.session_state["gif_bytes"]    = gif_bytes
        st.session_state["phase"]        = "risultato"
        st.rerun()

    # ══════════════════════════════════════════════════════════════
    #  PHASE 3 — RISULTATO
    # ══════════════════════════════════════════════════════════════
    elif phase == "risultato":
        image_data   = st.session_state.get("image_data")
        target_age   = st.session_state.get("target_age", 70)
        error        = st.session_state.get("result_error")
        result_bytes = st.session_state.get("result_bytes")
        gif_bytes    = st.session_state.get("gif_bytes")

        if error:
            st.error(f"Errore durante la trasformazione: {error}")
        else:
            card("RISULTATI TRASFORMAZIONE")
            col1, col2 = st.columns(2)
            with col1:
                mono("IMMAGINE ORIGINALE")
                st.image(image_data, use_container_width=True)
            with col2:
                mono(f"ETÀ TRASFORMATA: {target_age} ANNI", color="#00e5ff")
                st.image(result_bytes, use_container_width=True)
            card_end()

            _, dl_col, _ = st.columns([1, 2, 1])
            with dl_col:
                st.download_button(
                    label="⬇  SCARICA RISULTATO",
                    data=result_bytes,
                    file_name=f"age_transformed_{target_age}.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        _, reset_col, _ = st.columns([1, 2, 1])
        with reset_col:
            if st.button("↩  NUOVA ANALISI", use_container_width=True):
                for k in ["phase","image_data","target_age","result_bytes","result_error","gif_bytes"]:
                    st.session_state.pop(k, None)
                st.session_state["phase"] = "foto"
                st.rerun()