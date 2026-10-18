import streamlit as st
import streamlit.components.v1 as components
import tempfile
import os
import json
import base64
import boto3
import pandas as pd
import altair as alt
from dotenv import load_dotenv
from orchestrator import ModerationPipeline, CATEGORY_WEIGHTS

load_dotenv()
st.set_page_config(page_title="AI Censor Studio & StreamSafe OTT", layout="wide")

# Initialize AWS S3 Client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)
BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

@st.cache_resource
def load_ai_pipeline():
    return ModerationPipeline()

pipeline = load_ai_pipeline()

# ==========================================
# HELPER FUNCTIONS (PDF & AWS S3 SYNC)
# ==========================================
def create_simple_pdf_bytes(title, lines):
    """Generates a valid downloadable PDF file without requiring external libraries."""
    safe_lines = [title, "=" * len(title), ""] + lines
    y = 760
    content_ops = ["BT", "/F1 10 Tf"]
    for raw_line in safe_lines:
        # Wrap long lines to fit page width
        wrapped = [raw_line[i:i+85] for i in range(0, max(1, len(raw_line)), 85)]
        for subline in wrapped:
            clean = subline.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            content_ops.append(f"1 0 0 1 40 {y} Tm ({clean}) Tj")
            y -= 14
            if y < 40:
                break
        if y < 40:
            break
    content_ops.append("ET")
    stream_data = "\n".join(content_ops).encode("latin-1", errors="replace")

    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> "
        b"/MediaBox [0 0 612 792] /Contents 5 0 R >> endobj\n"
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        + f"5 0 obj << /Length {len(stream_data)} >> stream\n".encode("ascii")
        + stream_data
        + b"\nendstream endobj\n"
        b"xref\n0 6\n0000000000 65535 f \n"
        b"trailer << /Size 6 /Root 1 0 R >>\n%%EOF"
    )
    return pdf

def get_presigned_url(s3_key):
    try:
        return s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': BUCKET_NAME, 'Key': s3_key},
            ExpiresIn=7200
        )
    except Exception:
        return ""

def fetch_all_s3_movies():
    """Fetches all processed movies and their JSON manifests from AWS S3."""
    movies = []
    try:
        resp = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix="manifests/")
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if key.endswith("_manifest.json"):
                body = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)["Body"].read().decode("utf-8")
                manifest = json.loads(body)
                s3_video_key = f"processed/{manifest['file']}"
                manifest["stream_url"] = get_presigned_url(s3_video_key)
                movies.append(manifest)
    except Exception as e:
        st.sidebar.warning(f"Could not list S3 manifests: {e}")
    return movies

def render_html5_player(stream_url, manifest, kids_safe_enabled):
    """Renders HTML5 video player with English subtitles and live Kids-Safe skip/mute enforcement."""
    vtt_b64 = base64.b64encode(manifest.get("vtt_subtitles", "WEBVTT\n\n").encode("utf-8")).decode("utf-8")
    vtt_uri = f"data:text/vtt;base64,{vtt_b64}"

    js_payload = json.dumps({
        "skips": manifest.get("skip_intervals", []),
        "mutes": manifest.get("mute_intervals", [])
    })

    player_html = f"""
    <div style="background:#000; border-radius:10px; overflow:hidden; border:1px solid #333;">
        <video id="aiPlayer" width="100%" height="480" controls crossorigin="anonymous">
            <source src="{stream_url}" type="video/mp4">
            <track label="English Subtitles" kind="subtitles" srclang="en" src="{vtt_uri}" default>
        </video>
    </div>
    <script>
        const video = document.getElementById('aiPlayer');
        const data = {js_payload};
        const kidsSafe = {'true' if kids_safe_enabled else 'false'};
        const tracks = video.textTracks;

        video.addEventListener('timeupdate', function() {{
            const t = video.currentTime;

            if (!kidsSafe) {{
                // Kids mode OFF: never cut down scenes or force mute
                if (video.dataset.autoMuted === "true") {{
                    video.muted = false;
                    video.dataset.autoMuted = "false";
                }}
                if (tracks && tracks.length > 0) tracks[0].mode = 'showing';
                return;
            }}

            // 1. Kids mode ON: Instantly cut down / skip Violence, Weapons, Blood/Gore, Smoking, Nudity
            for (let i = 0; i < data.skips.length; i++) {{
                const seg = data.skips[i];
                if (t >= seg.start && t < seg.end) {{
                    video.currentTime = seg.end + 0.15;
                    return;
                }}
            }}

            // 2. Kids mode ON: Mute abusive words AND hide abusive subtitle during that moment
            let muteNow = false;
            for (let i = 0; i < data.mutes.length; i++) {{
                const seg = data.mutes[i];
                if (t >= seg.start && t <= seg.end) {{
                    muteNow = true;
                    break;
                }}
            }}

            video.muted = muteNow;
            video.dataset.autoMuted = muteNow ? "true" : "false";
            if (tracks && tracks.length > 0) {{
                tracks[0].mode = muteNow ? 'hidden' : 'showing';
            }}
        }});
    </script>
    """
    components.html(player_html, height=500)

# Load S3 catalog into session state
if "s3_catalog" not in st.session_state or not st.session_state.s3_catalog:
    st.session_state.s3_catalog = fetch_all_s3_movies()

if "recently_watched" not in st.session_state:
    # Populate Recently Watched with all S3 movies initially
    st.session_state.recently_watched = list(st.session_state.s3_catalog)

if "active_stream_movie" not in st.session_state:
    st.session_state.active_stream_movie = st.session_state.s3_catalog[0] if st.session_state.s3_catalog else None

# ==========================================
# LEFT SIDEBAR NAVIGATION & S3 CONTROLS
# ==========================================
st.sidebar.title("AI Censor Platform")
slide_choice = st.sidebar.radio(
    "Select Slide:",
    ["1. Main Slide (Censor Studio Dashboard)", "2. Second Slide (Kids-Safe Netflix Player)"]
)
st.sidebar.markdown("---")

if st.sidebar.button("Refresh Movies from AWS S3"):
    st.session_state.s3_catalog = fetch_all_s3_movies()
    st.session_state.recently_watched = list(st.session_state.s3_catalog)
    st.rerun()

# Left-hand slide down bars required for Main Slide
movie_titles = [m["title"] for m in st.session_state.s3_catalog]
selected_title = st.sidebar.selectbox(
    "Pick a Movie from AWS S3:",
    movie_titles if movie_titles else ["No movies in S3 yet"]
)

# Target Certificate dropdown (U and UA, with U as default)
target_cert = st.sidebar.selectbox(
    "Target Certificate:",
    options=["U", "UA"],
    index=0  # Default is "U"
)

# ==========================================
# SLIDE 1: MAIN SLIDE (CENSOR STUDIO DASHBOARD)
# ==========================================
if slide_choice == "1. Main Slide (Censor Studio Dashboard)":
    st.title("Main Slide: Censor Studio Dashboard")

    # --- 1. UPLOAD NEW MOVIE SECTION ---
    with st.expander("Upload & Screen a New Movie (Auto-Saves to AWS S3)", expanded=not bool(st.session_state.s3_catalog)):
        uploaded_file = st.file_uploader("Upload Movie (.mp4)", type=["mp4"])
        if uploaded_file is not None:
            if st.button("Run AI Screening & Upload to S3", type="primary"):
                file_name = uploaded_file.name
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(uploaded_file.read())
                    local_video_path = tmp.name

                with st.spinner("Running Whisper (English Subtitles), YOLOv8 (Bounding Boxes), and LLM Scoring..."):
                    manifest, annotated_video_path = pipeline.generate_censor_manifest(local_video_path)

                with st.spinner("Uploading Boxed Video & JSON Manifest to AWS S3..."):
                    s3_video_key = f"processed/{file_name}"
                    s3_manifest_key = f"manifests/{os.path.splitext(file_name)[0]}_manifest.json"
                    try:
                        with open(annotated_video_path, "rb") as vf:
                            s3_client.upload_fileobj(vf, BUCKET_NAME, s3_video_key)
                        s3_client.put_object(
                            Bucket=BUCKET_NAME,
                            Key=s3_manifest_key,
                            Body=json.dumps(manifest, indent=2),
                            ContentType="application/json"
                        )
                        st.success(f"Uploaded `{manifest['title']}` and JSON manifest to AWS S3!")
                    except Exception as e:
                        st.error(f"AWS S3 upload error: {e}")

                # Cleanup temp files & refresh S3 state
                for p in [local_video_path, annotated_video_path]:
                    if os.path.exists(p):
                        os.remove(p)

                st.session_state.s3_catalog = fetch_all_s3_movies()
                st.session_state.recently_watched = list(st.session_state.s3_catalog)
                st.session_state.active_stream_movie = st.session_state.s3_catalog[-1] if st.session_state.s3_catalog else None
                st.rerun()

    # Fetch selected movie from S3 catalog
    current_movie = next((m for m in st.session_state.s3_catalog if m["title"] == selected_title), None)

    if current_movie:
        st.subheader(f"Movie Screening: {current_movie['title']}")
        # Main Slide plays with Kids Mode OFF so censor officers see all bounding boxes on weapons/blood
        render_html5_player(current_movie["stream_url"], current_movie, kids_safe_enabled=False)

        # --- 2. VISUAL TIMELINE (BOXES FOR DETECTED EVENTS) ---
        st.markdown("---")
        st.subheader("Time Line (Detected Weapons, Blood/Gore, Violence, Smoking & Abusive Words)")
        events = current_movie.get("timeline_events", [])
        if events:
            df_tl = pd.DataFrame(events)
            tl_chart = alt.Chart(df_tl).mark_bar(cornerRadius=4, height=22).encode(
                x=alt.X("start:Q", title="Timeline (Seconds)"),
                x2="end:Q",
                y=alt.Y(
                    "category:N",
                    title="Detected Category",
                    sort=["Nudity", "Gore", "Strong abuse", "Violence", "Smoking or Drugs", "Mild Language"]
                ),
                color=alt.Color("category:N", legend=alt.Legend(title="Category")),
                tooltip=["category", "detail", "start", "end", "score"]
            ).properties(height=220)
            st.altair_chart(tl_chart, use_container_width=True)
        else:
            st.success("Clean Timeline: No weapons, blood, smoking, violence, or abusive words detected.")

        # --- 3. TIMELINE-BASED REPORT BOX ---
        st.markdown("---")
        rep_col1, rep_col2 = st.columns([4, 1])
        with rep_col1:
            st.subheader("Report (Based on Time Line)")
            with st.container(border=True):
                for line in current_movie.get("report_lines", []):
                    st.markdown(f"- {line}")
        with rep_col2:
            report_pdf = create_simple_pdf_bytes(
                f"TIMELINE DETECTION REPORT - {current_movie['title']}",
                current_movie.get("report_lines", [])
            )
            st.download_button(
                "⬇ Download Report as PDF",
                data=report_pdf,
                file_name=f"{current_movie['title']}_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        # --- 4. LLM CERTIFICATE BOX ---
        st.markdown("---")
        cert_col1, cert_col2 = st.columns([4, 1])
        issued_cert = current_movie["censor_rating"]
        total_pts = current_movie["total_severity_score"]

        with cert_col1:
            st.subheader("LLM Certificate (Based on the Report)")
            with st.container(border=True):
                # Display Metadata
                st.markdown(f"### Movie Title: **{current_movie['title']}**")
                st.markdown(f"**Synopsis / Description:** {current_movie['description']}")
                st.markdown(
                    f"**Duration:** `{current_movie['duration_str']}` | "
                    f"**Pixel Quality:** `{current_movie['pixel_quality']}` | "
                    f"**One-Time Severity Score:** `{total_pts} points` | "
                    f"**LLM Issued Certificate:** **`{issued_cert}`** *(User Target: `{target_cert}`)*"
                )

                # Display Criteria & One-Time Category Weights
                with st.expander("View Official One-Time Scoring Weights & Certificate Criteria", expanded=True):
                    w_col, c_col = st.columns(2)
                    with w_col:
                        st.markdown("**One-Time Category Weights (Max 1 allotment per movie):**")
                        for cat_name, w_val in CATEGORY_WEIGHTS.items():
                            allotted = current_movie["category_breakdown"].get(cat_name, 0)
                            status = f"Allotted: {allotted} pt" if allotted > 0 else "0 pt"
                            st.markdown(f"- **{cat_name}:** Weight `{w_val}` → *{status}*")
                    with c_col:
                        st.markdown("**Certificate Criteria:**")
                        st.markdown(
                            "- **U Certificate:** Score `≤ 4` AND **No** Nudity, **No** Strong Abusive Words, **No** Smoking/Drugs.\n"
                            "- **UA Certificate:** Score `5 to 11` AND **No** Nudity, **No** Strong Abusive Words.\n"
                            "- **A Certificate:** Score `≥ 12` OR contains any Nudity or Strong Abusive Words."
                        )

                # LLM Review ("Why", "How", "For What")
                st.info(f"**LLM Review (Why, How & For What):**\n\n{current_movie['llm_review']}")

                # Target Certificate vs LLM Certificate Comparison Logic
                if target_cert == issued_cert:
                    st.success(
                        f"**Target Matched ({target_cert} == {issued_cert}):** "
                        f"Your target certificate matches the LLM Issued Certificate. No scene cuts or suggestions are needed."
                    )
                else:
                    st.warning(
                        f"**Target Mismatch (Target: `{target_cert}` vs LLM Issued: `{issued_cert}`):** "
                        f"See the LLM's timestamped suggestions for the Movie Director / Production Team below:"
                    )
                    st.markdown("#### LLM Suggestions for Director / Production Team (When & Where):")
                    if issued_cert == "A":
                        st.markdown(f"- **Option A (For UA Certificate):** {current_movie['suggestion_for_ua']}")
                        if target_cert == "U":
                            st.markdown(f"- **Option B (For U Certificate):** {current_movie['suggestion_for_u']}")
                    elif issued_cert == "UA" and target_cert == "U":
                        st.markdown(f"- **Suggestion for U Certificate:** {current_movie['suggestion_for_u']}")
                    else:
                        st.markdown(f"- Movie already qualifies for `{issued_cert}`, which is even more permissive than `{target_cert}`.")

        with cert_col2:
            cert_pdf_lines = [
                f"Movie Title: {current_movie['title']}",
                f"Duration: {current_movie['duration_str']}",
                f"Pixel Quality: {current_movie['pixel_quality']}",
                f"Description: {current_movie['description']}",
                f"One-Time Severity Score: {total_pts}",
                f"LLM Issued Certificate: {issued_cert}",
                f"Target Certificate: {target_cert}",
                "",
                "LLM REVIEW (WHY / HOW / FOR WHAT):",
                current_movie['llm_review'],
            ]
            if target_cert != issued_cert:
                cert_pdf_lines += [
                    "",
                    "DIRECTOR / PRODUCTION TEAM SUGGESTIONS:",
                    current_movie.get("suggestion_for_ua", ""),
                    current_movie.get("suggestion_for_u", "")
                ]
            cert_pdf = create_simple_pdf_bytes("OFFICIAL AI CENSOR CERTIFICATE", cert_pdf_lines)
            st.download_button(
                "⬇ Download Certificate as PDF",
                data=cert_pdf,
                file_name=f"{current_movie['title']}_Certificate.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    else:
        st.info("No movies in S3 yet. Use the upload box above to screen your first movie!")


# ==========================================
# SLIDE 2: SECOND SLIDE (KIDS-SAFE NETFLIX UI)
# ==========================================
elif slide_choice == "2. Second Slide (Kids-Safe Netflix Player)":
    # Clean Layout: Only Video Player, Kids-Safe Toggle, Metadata, and 4 Netflix Rows
    head_col1, head_col2 = st.columns([3, 1])
    with head_col1:
        st.title("StreamSafe Cinema")
    with head_col2:
        st.write("")
        kids_safe_toggle = st.toggle("Button for Kids Safety", value=True)

    active_movie = st.session_state.active_stream_movie
    if active_movie:
        if kids_safe_toggle:
            st.success("**Kids Safety ON**: Violence, weapons, blood, and smoking scenes cut down instantly. Abusive words/subtitles are muted.")
        else:
            st.warning("**Kids Safety OFF**: Playing unedited movie with weapon/blood bounding boxes and English subtitles.")

        render_html5_player(active_movie["stream_url"], active_movie, kids_safe_enabled=kids_safe_toggle)

        # Certificate Badge + Description on the next line below the movie
        cert_badge = active_movie["censor_rating"]
        badge_bg = "#28a745" if cert_badge == "U" else ("#ffc107" if cert_badge == "UA" else "#dc3545")
        st.markdown(
            f"### {active_movie['title']} &nbsp; "
            f"<span style='background:{badge_bg}; color:#000; padding:3px 12px; border-radius:6px; font-weight:bold;'>"
            f"Certificate: {cert_badge}</span>",
            unsafe_allow_html=True
        )
        st.write(active_movie["description"])
        st.markdown("---")

    # --- 4 NETFLIX CATEGORIES ---
    def render_netflix_row(row_title, movies_list, row_key):
        st.subheader(row_title)
        if not movies_list:
            st.caption("No movies in this category yet.")
            return
        cols = st.columns(4)
        for idx, mov in enumerate(movies_list):
            with cols[idx % 4]:
                with st.container(border=True):
                    st.markdown(f"** {mov['title']}**")
                    st.caption(f"Certificate: **{mov['censor_rating']}** | {mov['duration_str']} | {mov['pixel_quality']}")
                    st.caption(mov["description"][:90] + "...")
                    if st.button("Watch Movie", key=f"play_{row_key}_{mov['file']}_{idx}", use_container_width=True):
                        st.session_state.active_stream_movie = mov
                        # Ensure watched movie is placed at the front of Recently Watched as well as its category
                        st.session_state.recently_watched = [
                            m for m in st.session_state.recently_watched if m["file"] != mov["file"]
                        ]
                        st.session_state.recently_watched.insert(0, mov)
                        st.rerun()

    u_movies = [m for m in st.session_state.s3_catalog if m["censor_rating"] == "U"]
    ua_movies = [m for m in st.session_state.s3_catalog if m["censor_rating"] == "UA"]
    a_movies = [m for m in st.session_state.s3_catalog if m["censor_rating"] == "A"]

    # 1. Recently Watched (Fetches all S3 movies and updates when user plays any movie)
    render_netflix_row("Recently Watched", st.session_state.recently_watched, "recent")
    # 2. Kids and Family for U
    render_netflix_row("Kids and Family (U)", u_movies, "cat_u")
    # 3. Action & Thriller for UA
    render_netflix_row("Action & Thriller (UA)", ua_movies, "cat_ua")
    # 4. Mature & Restricted for A
    render_netflix_row("Mature & Restricted (A)", a_movies, "cat_a")
