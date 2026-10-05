"""Streamlit demo for HAM10000 lesion classification."""

import hashlib
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from predict_hybrid import (
    CLASS_FULL_NAMES,
    inspect_image,
    load_models,
    predict_image,
)


st.set_page_config(
    page_title="SkinScope | Phân tích ảnh da",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --ink:#13263b; --muted:#617287; --teal:#087e80; --mint:#e9f6f3; --line:#e3ebef; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--ink); }
    .stApp { background:linear-gradient(180deg,#f5faf9 0%,#f7f9fc 52%,#f4f7fa 100%); }
    .block-container { max-width:1280px; padding-top:2rem; padding-bottom:3rem; }
    h1,h2,h3 { font-family:'Manrope',sans-serif; letter-spacing:-.035em; }
    .hero { padding:2.1rem 2.2rem; border:1px solid #d9eae7; border-radius:24px;
      background:radial-gradient(circle at 90% 5%,#d8f0e8 0,transparent 32%),linear-gradient(115deg,#fff,#eef8f6 68%,#eff5fb);
      box-shadow:0 14px 40px rgba(32,71,82,.07); margin-bottom:1.5rem; }
    .eyebrow { color:var(--teal); font-size:.78rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
    .hero h1 { margin:.55rem 0 .5rem; font-size:2.25rem; }
    .hero p { color:var(--muted); max-width:760px; font-size:1.02rem; margin:0; line-height:1.65; }
    .card { background:white; border:1px solid var(--line); border-radius:20px; padding:1.35rem 1.5rem; box-shadow:0 8px 24px rgba(34,58,79,.045); }
    .section-label { color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.1em; font-weight:700; }
    .result-name { font-family:'Manrope',sans-serif; font-size:1.45rem; font-weight:800; color:var(--ink); margin:.35rem 0 .15rem; }
    .result-code { display:inline-block; background:var(--mint); color:var(--teal); border-radius:999px; padding:.25rem .7rem; font-weight:700; font-size:.78rem; }
    .small-note { color:var(--muted); font-size:.88rem; line-height:1.55; }
    div[data-testid="stFileUploader"] { background:#fff; border:1px dashed #9fcac1; border-radius:16px; padding:.6rem; }
    div.stButton > button[kind="primary"] { background:linear-gradient(110deg,#087e80,#149b91); border:0; border-radius:12px; padding:.68rem 1.25rem; font-weight:700; }
    section[data-testid="stSidebar"] { background:#f1f7f6; border-right:1px solid var(--line); }
    div[data-testid="stMetric"] { background:#f5faf9; border:1px solid var(--line); padding:1rem; border-radius:14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Hệ thống hỗ trợ nghiên cứu · HAM10000</div>
      <h1>Phân tích tổn thương da</h1>
      <p>Tải ảnh dermoscopy để xem các lớp mô hình xếp hạng và vùng ảnh có ảnh hưởng
      đến dự đoán. Hệ thống kiểm tra ảnh trước, không tự ý cắt hoặc chỉnh sửa ảnh.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_models():
    return load_models()


try:
    with st.spinner("Đang tải các mô hình…"):
        cnn, vit, hybrid, device = get_models()
except Exception as exc:
    st.error(f"Không tải được mô hình: {exc}")
    st.info("Kiểm tra các checkpoint trong thư mục outputs rồi tải lại trang.")
    st.stop()

with st.sidebar:
    st.markdown("### SkinScope")
    st.caption("Bản demo phân loại ảnh dermoscopy")
    st.markdown("---")
    st.markdown("**Kiến trúc**")
    st.write("ResNet50 + ViT-B/16")
    st.write("Ghép đặc trưng: 2.816 chiều")
    st.write("Bộ phân loại: Weighted Cross Entropy")
    st.markdown("---")
    st.markdown("**Thiết bị**")
    st.write(str(device))
    st.caption("Grad-CAM được tạo từ nhánh ResNet50.")

st.markdown("### 1 · Chọn ảnh")
uploaded_file = st.file_uploader(
    "Tải ảnh dermoscopy lên",
    type=["jpg", "jpeg", "png"],
    help="Ảnh được giữ nguyên; hệ thống chỉ đổi kích thước để đưa vào mô hình.",
)

if uploaded_file is None:
    st.markdown(
        "<div class='card'><span class='section-label'>Bắt đầu</span><p class='small-note'>"
        "Chọn ảnh JPG hoặc PNG. Ảnh mẫu có thể lấy trong thư mục demo_samples của project.</p></div>",
        unsafe_allow_html=True,
    )
else:
    raw_bytes = uploaded_file.getvalue()
    image_key = hashlib.sha256(raw_bytes).hexdigest()
    try:
        image = Image.open(BytesIO(raw_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("Tệp tải lên không đọc được như ảnh. Hãy chọn JPG hoặc PNG hợp lệ.")
        st.stop()

    quality = inspect_image(image)
    left, right = st.columns([1.03, 0.97], gap="large")
    with left:
        st.markdown("### 2 · Kiểm tra ảnh")
        st.image(image, caption="Ảnh gốc — chưa bị cắt", use_container_width=True)
        st.caption(f"Kích thước: {quality['width']} × {quality['height']} px")
        if quality["has_warnings"]:
            for issue in quality["issues"]:
                st.warning(issue)
            for suggestion in quality["suggestions"]:
                st.caption("Gợi ý: " + suggestion)
            st.caption("Kiểm tra tự động chỉ là heuristic; vùng tối có thể là đặc điểm thật của ảnh.")
        else:
            st.success("Không phát hiện viền đen hoặc vùng trống rõ rệt theo kiểm tra tự động.")

    with right:
        st.markdown("### 3 · Phân tích")
        st.markdown(
            "<div class='card'><span class='section-label'>Trước khi tiếp tục</span>"
            "<p class='small-note'>Xem ảnh gốc và cảnh báo chất lượng. Nếu ảnh phù hợp, bấm nút để chạy mô hình.</p></div>",
            unsafe_allow_html=True,
        )
        if st.button("Phân tích ảnh", type="primary", use_container_width=True):
            try:
                with st.spinner("Đang phân tích và tạo bản đồ Grad-CAM…"):
                    result = predict_image(image, cnn, vit, hybrid, device)
                st.session_state["analysis"] = {"image_key": image_key, "result": result}
            except Exception as exc:
                st.error(f"Phân tích không thành công: {exc}")

    saved = st.session_state.get("analysis")
    if saved and saved.get("image_key") == image_key:
        result = saved["result"]
        st.markdown("---")
        st.markdown("### Kết quả mô hình")
        result_col, cam_col = st.columns([0.9, 1.1], gap="large")

        with result_col:
            first, second = result["ranked_classes"]
            st.markdown(
                f"<div class='card'><span class='section-label'>Xếp hạng 1</span>"
                f"<div style='margin:.55rem 0'><span class='result-code'>{first['class_code'].upper()}</span></div>"
                f"<div class='result-name'>{first['class_name']}</div></div>",
                unsafe_allow_html=True,
            )
            st.metric("Điểm softmax của lớp đứng đầu", f"{first['score'] * 100:.2f}%")
            st.markdown("#### Lớp gợi ý thứ hai")
            st.write(f"**{second['class_code'].upper()} · {second['class_name']}**")
            st.progress(float(second["score"]))
            st.caption(f"Điểm softmax: {second['score'] * 100:.2f}%")
            st.info(
                "Đây là hai lớp được xếp hạng cao nhất trong phân loại một nhãn. "
                "Lớp thứ hai là khả năng thay thế, không xác nhận bệnh thứ hai cùng xuất hiện."
            )
            st.warning(
                "Điểm softmax chưa được hiệu chỉnh xác suất; không diễn giải phần trăm này "
                "như xác suất chẩn đoán chính xác."
            )

        with cam_col:
            st.markdown("#### Vùng ảnh ảnh hưởng đến dự đoán")
            st.image(result["gradcam"], caption="Grad-CAM · nhánh ResNet50", use_container_width=True)
            st.caption(
                "Vùng tô màu cho biết khu vực có ảnh hưởng đến điểm của lớp dự đoán. "
                "Đây không phải phân đoạn hay đường viền y khoa của tổn thương."
            )

        st.markdown("#### Điểm của cả 7 lớp")
        scores = pd.DataFrame([
            {
                "Mã lớp": code.upper(),
                "Tên lớp": CLASS_FULL_NAMES[code],
                "Điểm softmax (%)": round(score * 100, 2),
            }
            for code, score in result["all_scores"].items()
        ]).sort_values("Điểm softmax (%)", ascending=False)
        st.dataframe(scores, hide_index=True, use_container_width=True)

st.markdown("---")
with st.expander("Các lớp trong HAM10000"):
    class_table = pd.DataFrame([
        {"Mã lớp": code.upper(), "Tên lớp": name}
        for code, name in CLASS_FULL_NAMES.items()
    ])
    st.dataframe(class_table, hide_index=True, use_container_width=True)

st.caption(
    "Công cụ minh họa phục vụ nghiên cứu. Kết quả không thay thế đánh giá của bác sĩ "
    "và không dùng làm chẩn đoán y khoa."
)
