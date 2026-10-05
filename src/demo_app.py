"""Polished Streamlit interface for the HAM10000 research demo."""

import hashlib
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from predict_hybrid import (
    CLASS_FULL_NAMES,
    inspect_image,
    load_models,
    predict_image,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DEMO_DIR = BASE_DIR / "demo_samples"

st.set_page_config(
    page_title="SkinScope · Dermoscopy AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
      --ink:#142b3b; --muted:#637b89; --teal:#087e80; --mint:#dff3ed;
      --line:#deebe8; --paper:#ffffff; --canvas:#f4f8f7; --blue:#eaf1f7;
    }
    html, body, [class*="css"] { font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink); }
    .stApp { background:radial-gradient(ellipse at 3% 0%,#e7f5f0 0%,transparent 28%),linear-gradient(180deg,#f8fbfa 0%,#f2f6f8 100%); }
    .block-container { max-width:1320px; padding:1.4rem 2.4rem 3.2rem; }
    #MainMenu, footer { visibility:hidden; }
    header[data-testid="stHeader"] { background:transparent; }
    h1,h2,h3,h4 { color:var(--ink); letter-spacing:-.035em; }
    .brandline { display:flex; align-items:center; gap:.65rem; color:var(--teal); font-weight:800; letter-spacing:.11em; font-size:.78rem; text-transform:uppercase; }
    .brandmark { width:30px; height:30px; display:inline-flex; justify-content:center; align-items:center; border-radius:10px; color:white; font-size:1.1rem; background:linear-gradient(145deg,#0b9690,#087e80); box-shadow:0 5px 12px #087e8033; }
    .hero { position:relative; overflow:hidden; margin:.8rem 0 1.5rem; padding:2.5rem 2.7rem; border:1px solid #d7e8e4; border-radius:28px; background:radial-gradient(circle at 91% 0%,#cdece2 0,transparent 30%),radial-gradient(circle at 77% 100%,#edf4fb 0,transparent 36%),linear-gradient(112deg,#fff 0%,#f4fbf8 70%,#f1f7fb 100%); box-shadow:0 18px 50px rgba(25,67,73,.075); }
    .hero-kicker { color:var(--teal); font-size:.77rem; letter-spacing:.13em; font-weight:800; text-transform:uppercase; }
    .hero h1 { font-size:clamp(2rem,4vw,3.15rem); margin:.48rem 0 .65rem; font-weight:800; line-height:1.08; }
    .hero p { color:var(--muted); max-width:760px; margin:0; line-height:1.72; font-size:1.03rem; }
    .pill-row { display:flex; gap:.55rem; flex-wrap:wrap; margin-top:1.25rem; }
    .pill { padding:.42rem .72rem; border:1px solid #d9e8e4; border-radius:999px; background:#ffffffb8; color:#34515f; font-size:.78rem; font-weight:650; }
    .step-head { display:flex; align-items:center; gap:.75rem; margin:.35rem 0 .85rem; }
    .step-num { width:30px; height:30px; border-radius:10px; display:inline-flex; align-items:center; justify-content:center; background:var(--mint); color:var(--teal); font-size:.84rem; font-weight:800; }
    .step-title { font-size:1.08rem; font-weight:780; letter-spacing:-.02em; color:var(--ink); }
    .surface { height:100%; background:var(--paper); border:1px solid var(--line); border-radius:22px; padding:1.2rem 1.35rem; box-shadow:0 10px 28px rgba(26,61,77,.045); }
    .surface-title { margin:0 0 .28rem; color:var(--ink); font-weight:760; font-size:1rem; }
    .surface-copy { margin:0; color:var(--muted); font-size:.87rem; line-height:1.6; }
    .section-eyebrow { color:var(--muted); text-transform:uppercase; letter-spacing:.1em; font-size:.72rem; font-weight:800; }
    .class-card { background:linear-gradient(145deg,#fff,#f5fbf9); border:1px solid var(--line); border-radius:20px; padding:1.25rem 1.35rem; min-height:150px; box-shadow:0 9px 24px rgba(26,61,77,.045); }
    .class-card.alt { background:linear-gradient(145deg,#fff,#f4f7fb); }
    .class-code { display:inline-flex; border-radius:999px; padding:.3rem .68rem; background:var(--mint); color:var(--teal); font-size:.74rem; font-weight:850; letter-spacing:.06em; }
    .alt .class-code { color:#44688b; background:var(--blue); }
    .class-name { margin:.75rem 0 .7rem; min-height:2.9rem; font-size:1.17rem; line-height:1.28; font-weight:800; letter-spacing:-.025em; }
    .score-line { display:flex; justify-content:space-between; gap:1rem; align-items:baseline; color:var(--muted); font-size:.82rem; }
    .score-line strong { color:var(--ink); font-size:1.2rem; }
    .note { padding:.85rem 1rem; border-radius:14px; background:#f0f7f6; color:#496572; font-size:.83rem; line-height:1.55; border:1px solid #e0eeeb; }
    .stButton > button[kind="primary"] { min-height:3rem; border:0; border-radius:13px; color:#fff; font-weight:750; background:linear-gradient(105deg,#087e80,#10a092); box-shadow:0 8px 18px #087e8033; transition:transform .15s ease,box-shadow .15s ease; }
    .stButton > button[kind="primary"]:hover { transform:translateY(-1px); box-shadow:0 11px 23px #087e8040; }
    div[data-testid="stFileUploader"] { padding:.55rem; border:1px dashed #a8cfc6; border-radius:16px; background:#fbfefd; }
    div[data-testid="stFileUploader"] section { border:0; }
    div[data-testid="stMetric"] { border:1px solid var(--line); border-radius:15px; background:#fff; padding:1rem 1.1rem; box-shadow:0 5px 16px #183b4d0a; }
    div[data-testid="stAlert"] { border-radius:14px; }
    div[data-testid="stTabs"] button { font-weight:700; }
    section[data-testid="stSidebar"] { background:#f1f7f6; border-right:1px solid var(--line); }
    .footer-note { margin-top:2.5rem; padding-top:1rem; border-top:1px solid var(--line); color:var(--muted); font-size:.8rem; line-height:1.6; }
    @media (max-width:760px) {
      .block-container { padding:.8rem 1rem 2.2rem; }
      .hero { padding:1.55rem 1.35rem; border-radius:21px; }
      .hero p { font-size:.94rem; }
      .surface { padding:1rem; }
      .class-card { min-height:0; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='brandline'><span class='brandmark'>◈</span> SkinScope <span style='color:#91a6ab;font-weight:600;letter-spacing:.02em'>/ HAM10000</span></div>",
    unsafe_allow_html=True,
)
st.markdown(
    """
    <section class="hero">
      <div class="hero-kicker">Dermoscopy research workspace</div>
      <h1>Hiểu rõ hơn điều mô hình nhìn thấy.</h1>
      <p>Tải ảnh tổn thương da, kiểm tra chất lượng đầu vào và xem các lớp được xếp hạng
      cùng bản đồ Grad-CAM. Ảnh gốc luôn được giữ nguyên trong quá trình kiểm tra.</p>
      <div class="pill-row"><span class="pill">ResNet50</span><span class="pill">Vision Transformer</span><span class="pill">7 lớp HAM10000</span><span class="pill">Grad-CAM</span></div>
    </section>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_models():
    return load_models()


with st.sidebar:
    st.markdown("## ◈ SkinScope")
    st.caption("Bản demo hỗ trợ nghiên cứu ảnh dermoscopy")
    st.markdown("---")
    st.markdown("**Mô hình**")
    st.write("ResNet50 + ViT-B/16")
    st.write("Ghép đặc trưng 2.816 chiều")
    st.write("Weighted Cross Entropy")
    st.markdown("---")
    st.markdown("**Giải thích kết quả**")
    st.caption("Grad-CAM minh họa vùng ảnh ảnh hưởng đến điểm dự đoán; không phải phân đoạn tổn thương.")
    st.caption("Điểm softmax chưa được hiệu chỉnh thành xác suất lâm sàng.")


def step_heading(number, title):
    st.markdown(
        f"<div class='step-head'><span class='step-num'>{number}</span><span class='step-title'>{title}</span></div>",
        unsafe_allow_html=True,
    )


step_heading("01", "Chọn ảnh dermoscopy")
source = st.radio(
    "Nguồn ảnh",
    ["Tải ảnh lên", "Thử ảnh mẫu"],
    horizontal=True,
    label_visibility="collapsed",
)

image = None
raw_bytes = None
display_name = None
reference_label = None

if source == "Tải ảnh lên":
    uploaded_file = st.file_uploader(
        "Kéo ảnh vào đây hoặc chọn từ thiết bị",
        type=["jpg", "jpeg", "png"],
        help="JPG hoặc PNG. Ảnh sẽ không bị tự động cắt.",
    )
    if uploaded_file is not None:
        raw_bytes = uploaded_file.getvalue()
        display_name = uploaded_file.name
else:
    sample_files = sorted(path.name for path in DEMO_DIR.glob("*.jpg")) if DEMO_DIR.exists() else []
    if sample_files:
        selected_sample = st.selectbox(
            "Chọn ảnh mẫu",
            sample_files,
            format_func=lambda name: Path(name).stem.replace("_", " · "),
        )
        raw_bytes = (DEMO_DIR / selected_sample).read_bytes()
        display_name = selected_sample
        reference_label = Path(selected_sample).stem.split("_")[0].upper()
    else:
        st.info("Không tìm thấy ảnh trong demo_samples.")

if raw_bytes is not None:
    image_key = hashlib.sha256(raw_bytes).hexdigest()
    try:
        image = Image.open(BytesIO(raw_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("Không đọc được tệp này như ảnh JPG hoặc PNG hợp lệ.")

if image is None:
    st.markdown(
        "<div class='surface'><p class='surface-title'>Bắt đầu nhanh</p><p class='surface-copy'>"
        "Tải ảnh của bạn lên hoặc chọn một trong các ảnh mẫu HAM10000 để xem giao diện phân tích.</p></div>",
        unsafe_allow_html=True,
    )
else:
    quality = inspect_image(image)
    preview_col, quality_col = st.columns([1.12, 0.88], gap="large")
    with preview_col:
        st.markdown("<div class='section-eyebrow'>Ảnh xem trước</div>", unsafe_allow_html=True)
        st.image(image, caption=display_name, use_container_width=True)
    with quality_col:
        st.markdown("<div class='section-eyebrow'>Kiểm tra đầu vào</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='surface'><p class='surface-title'>{quality['width']} × {quality['height']} px</p>"
            "<p class='surface-copy'>Ảnh được chuyển về kích thước mô hình khi dự đoán; tệp gốc không bị cắt.</p></div>",
            unsafe_allow_html=True,
        )
        if reference_label:
            st.caption(f"Nhãn tham chiếu của ảnh mẫu: {reference_label}")
        if quality["has_warnings"]:
            for issue in quality["issues"]:
                st.warning(issue)
            for suggestion in quality["suggestions"]:
                st.caption("Gợi ý: " + suggestion)
            st.caption("Kiểm tra viền là heuristic; vùng tối cũng có thể thuộc tổn thương thật.")
        else:
            st.success("Không phát hiện viền đen hoặc vùng đồng màu đáng kể.")

    step_heading("02", "Chạy phân tích")
    st.markdown(
        "<div class='note'>Mô hình xếp hạng 7 lớp trong HAM10000. Hãy xem ảnh và cảnh báo chất lượng trước khi chạy.</div>",
        unsafe_allow_html=True,
    )
    button_col, hint_col = st.columns([.42, .58], vertical_alignment="center")
    with button_col:
        analyze_clicked = st.button("✦  Phân tích ảnh", type="primary", use_container_width=True)
    with hint_col:
        st.caption("Lần phân tích đầu có thể mất thêm thời gian để tải các checkpoint.")

    if analyze_clicked:
        try:
            with st.spinner("Đang tải mô hình và phân tích ảnh…"):
                model_bundle = get_models()
                result = predict_image(image, *model_bundle)
            st.session_state["analysis"] = {"image_key": image_key, "result": result}
        except Exception as exc:
            st.error(f"Phân tích chưa hoàn tất: {exc}")

    saved = st.session_state.get("analysis")
    if saved and saved.get("image_key") == image_key:
        result = saved["result"]
        st.markdown("---")
        step_heading("03", "So sánh cách ba mô hình phân tích")
        model_info = [
            ("ResNet50", "CNN · nhận dạng đặc trưng cục bộ như màu sắc, đường viền và kết cấu", ""),
            ("ViT-B/16", "Transformer · xem quan hệ giữa các mảng ảnh trên phạm vi toàn ảnh", "alt"),
            ("Hybrid", "Kết hợp vector đặc trưng ResNet50 và ViT-B/16 để đưa ra dự đoán cuối", ""),
        ]
        model_columns = st.columns(3, gap="medium")
        for column, (model_name, description, style) in zip(model_columns, model_info):
            ranked = result["model_rankings"][model_name]
            best, runner_up = ranked[0], ranked[1]
            with column:
                st.markdown(
                    f"<div class='class-card {style}'><div class='section-eyebrow'>{model_name}</div>"
                    f"<div class='class-name'>{best['class_code'].upper()} · {best['class_name']}</div>"
                    f"<div class='score-line'><span>Điểm lớp đứng đầu</span><strong>{best['score'] * 100:.1f}%</strong></div>"
                    f"<div class='surface-copy' style='margin-top:.8rem'>{description}</div>"
                    f"<div class='surface-copy' style='margin-top:.65rem'>Hạng 2: {runner_up['class_code'].upper()} · {runner_up['score'] * 100:.1f}%</div></div>",
                    unsafe_allow_html=True,
                )

        winners = [result["model_rankings"][name][0]["class_code"] for name, _, _ in model_info]
        if len(set(winners)) == 1:
            st.success(f"Cả ba mô hình cùng xếp {winners[0].upper()} ở vị trí đầu.")
        else:
            st.info("Các mô hình chưa đồng thuận lớp đứng đầu. Hãy xem điểm từng lớp và Grad-CAM để so sánh; bất đồng không tự xác định mô hình nào đúng.")
        st.info(
            "Mỗi ảnh HAM10000 chỉ có một nhãn. Các lớp hạng 2 là phương án dự đoán khác, "
            "không xác nhận có bệnh thứ hai cùng xuất hiện."
        )
        st.warning("Điểm softmax chưa được hiệu chỉnh; không xem đây là xác suất chẩn đoán chính xác.")

        resnet_tab, hybrid_tab, scores_tab = st.tabs(["Giải thích ResNet50", "Giải thích Hybrid", "So sánh điểm 7 lớp"])
        with resnet_tab:
            original_tab, cam_tab = st.tabs(["Ảnh gốc", "Grad-CAM ResNet50"])
            with original_tab:
                st.image(image, use_container_width=True)
            with cam_tab:
                st.image(result["gradcam_resnet"], use_container_width=True)
                st.caption("Grad-CAM cho lớp ResNet50 xếp đầu; vùng tô màu minh họa ảnh hưởng, không phải vùng bệnh được phân đoạn.")
            st.markdown("**Cách đọc:** ResNet50 xử lý ảnh bằng các lớp tích chập, thường nhạy với hoa văn và đặc trưng cục bộ. Bản đồ Grad-CAM cho thấy khu vực ảnh hưởng đến điểm của lớp đứng đầu.")
        with hybrid_tab:
            original_hybrid, cam_hybrid = st.tabs(["Ảnh gốc", "Grad-CAM Hybrid"])
            with original_hybrid:
                st.image(image, use_container_width=True)
            with cam_hybrid:
                st.image(result["gradcam_hybrid"], use_container_width=True)
                st.caption("Đây là gradient của đầu phân loại Hybrid theo đặc trưng ResNet50; ViT không có bản đồ chú ý riêng trong demo này.")
            st.markdown("**Cách đọc:** Hybrid nối vector ResNet50 (2048 chiều) và ViT (768 chiều), rồi phân loại trên vector kết hợp 2816 chiều. Grad-CAM ở đây chỉ minh họa ảnh hưởng qua nhánh ResNet50.")
        with scores_tab:
            score_frame = pd.DataFrame([
                {
                    "Mã lớp": code.upper(),
                    "Tên lớp": CLASS_FULL_NAMES[code],
                    "ResNet50 (%)": round(result["all_scores"]["ResNet50"][code] * 100, 2),
                    "ViT-B/16 (%)": round(result["all_scores"]["ViT-B/16"][code] * 100, 2),
                    "Hybrid (%)": round(result["all_scores"]["Hybrid"][code] * 100, 2),
                }
                for code in CLASS_FULL_NAMES
            ])
            chart_data = score_frame.set_index("Mã lớp")[["ResNet50 (%)", "ViT-B/16 (%)", "Hybrid (%)"]]
            st.bar_chart(chart_data, horizontal=True)
            with st.expander("Xem bảng điểm đầy đủ"):
                st.dataframe(score_frame, hide_index=True, use_container_width=True)

with st.expander("Tìm hiểu mô hình và 7 lớp"):
    st.markdown(
        "Mô hình ghép đặc trưng của ResNet50 và ViT-B/16 rồi xếp hạng bảy nhóm tổn thương của HAM10000. "
        "Điểm hiển thị là softmax thô, chưa được hiệu chỉnh xác suất."
    )
    st.dataframe(
        pd.DataFrame([{"Mã lớp": code.upper(), "Tên lớp": name} for code, name in CLASS_FULL_NAMES.items()]),
        hide_index=True,
        use_container_width=True,
    )

st.markdown(
    "<div class='footer-note'><b>Lưu ý:</b> SkinScope là bản minh họa nghiên cứu, không phải thiết bị y tế. "
    "Kết quả không thay thế đánh giá của bác sĩ và không dùng để tự chẩn đoán hoặc điều trị.</div>",
    unsafe_allow_html=True,
)
