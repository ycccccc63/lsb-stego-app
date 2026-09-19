import streamlit as st
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

# 頁面配置
st.set_page_config(page_title="LSB 影像隱寫嵌入系統", layout="wide")

# 自訂 CSS 還原設計樣式
st.markdown("""
<style>
    .main-header {
        background-color: #1E2238;
        color: white;
        padding: 20px 25px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .main-header h2 { color: white; margin: 0; font-size: 24px; }
    .main-header p { color: #94A3B8; margin: 5px 0 0 0; font-size: 14px; }
    .block-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- 核心演算法 -----------------

def text_to_bits(text):
    data = text.encode('utf-8')
    length = len(data)
    len_bits = format(length, '032b') # 32 bits 記錄位元組長度
    text_bits = ''.join(format(b, '08b') for b in data)
    return [int(b) for b in (len_bits + text_bits)]

def embed_lsb(img, text, ratio):
    # 確保轉為 RGB 格式
    arr = np.array(img.convert('RGB'), dtype=np.uint8)
    h, w, c = arr.shape
    total_pixels = h * w
    
    selected_pixels = max(1, int(total_pixels * (ratio / 100.0)))
    capacity_bits = selected_pixels * 3
    secret_bits = text_to_bits(text)
    
    if len(secret_bits) > capacity_bits:
        raise ValueError(f"嵌入容量不足：目前比例僅能容納 {capacity_bits} bits，但訊息需 {len(secret_bits)} bits。請提高比例！")
        
    flat_arr = arr.reshape(-1, 3).copy()
    bit_idx = 0
    total_bits = len(secret_bits)
    
    for i in range(selected_pixels):
        for ch in range(3):
            if bit_idx < total_bits:
                # 使用 & 254 (0b11111110) 取代 & ~1，避免 uint8 負數溢位
                flat_arr[i, ch] = (int(flat_arr[i, ch]) & 254) | int(secret_bits[bit_idx])
                bit_idx += 1
            else:
                break
        if bit_idx >= total_bits:
            break
            
    stego_arr = flat_arr.reshape(h, w, 3)
    return Image.fromarray(stego_arr.astype(np.uint8))

def chi_square_attack(img):
    arr = np.array(img.convert('L')).flatten()
    counts = np.bincount(arr, minlength=256)
    
    # 觀察相鄰值對 (2k, 2k+1)
    observed_even = counts[0::2]
    observed_odd = counts[1::2]
    expected = (observed_even + observed_odd) / 2.0
    
    mask = expected > 0
    chi2 = np.sum(((observed_even[mask] - expected[mask])**2) / expected[mask])
    
    k = np.sum(mask) - 1
    if k <= 0:
        return 1.0
    z = (chi2 - k) / np.sqrt(2 * k)
    p_val = 0.5 * (1.0 - float(np.tanh(z / np.sqrt(2))))
    return max(0.0, min(1.0, p_val))

# ----------------- 介面排版 -----------------

# 標題區
st.markdown("""
<div class="main-header">
    <h2>LSB 影像隱寫嵌入系統</h2>
    <p>最低有效位元替換法 (LSB Substitution) 教學實作介面</p>
</div>
""", unsafe_allow_html=True)

# 1. 欲嵌入的訊息
st.markdown("#### ❶ 欲嵌入的訊息 <span style='font-size:13px; color:#64748B; font-weight:normal;'>使用者輸入想藏入影像中的文字內容</span>", unsafe_allow_html=True)
secret_text = st.text_input("輸入欲嵌入文字：", value="nknu0123456789", label_visibility="collapsed")
bits_needed = 32 + len(secret_text.encode('utf-8')) * 8
st.caption(f"已輸入 {len(secret_text)} 個字元 | 約 {bits_needed} bits 容量")

st.write("")

# 2 & 4. 影像輸入與輸出（雙欄排版）
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### ❷ 檢入影像 <span style='font-size:13px; color:#64748B; font-weight:normal;'>支援 PNG / BMP 格式</span>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("拖曳或點擊選擇影像", type=["png", "bmp"], label_visibility="collapsed")
    
    if uploaded_file:
        input_img = Image.open(uploaded_file)
        st.image(input_img, caption=f"已選擇：{uploaded_file.name}", use_container_width=True)
    else:
        st.info("請點擊上方區塊上傳一張 PNG 或 BMP 影像檔案。")
        input_img = None

with col_right:
    st.markdown("#### ❹ 輸出：LSB 藏入後影像 <span style='font-size:13px; color:#64748B; font-weight:normal;'>處理完成後可預覽並下載</span>", unsafe_allow_html=True)
    output_container = st.empty()
    download_btn_container = st.empty()
    output_container.info("尚未執行嵌入處理。")

st.write("")

# 3. 比例滑桿與執行
st.markdown("#### ❸ LSB 嵌入的像素之比例 <span style='font-size:13px; color:#64748B; font-weight:normal;'>可自選，避免被統計分析偵測</span>", unsafe_allow_html=True)
ratio = st.slider("比例 (%)", min_value=1, max_value=100, value=27, step=1, label_visibility="collapsed")
st.caption(f"已選取影像中 **{ratio}%** 的像素進行 LSB 替換。比例越低，被卡方檢定等統計方法偵測到的機率越低。")

run_btn = st.button("🔒 執行 LSB 嵌入", type="primary", use_container_width=True)

# 執行邏輯
stego_img = None
if run_btn:
    if not input_img:
        st.error("請先在步驟 ❷ 上傳影像！")
    elif not secret_text:
        st.error("請在步驟 ❶ 輸入嵌入文字！")
    else:
        try:
            stego_img = embed_lsb(input_img, secret_text, ratio)
            
            # 在步驟 ❹ 顯示產生的影像與下載按鈕
            output_container.image(stego_img, caption="stego_output.png (已完成嵌入)", use_container_width=True)
            
            # 轉換為 byte buffer 供下載
            buf = io.BytesIO()
            stego_img.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            download_btn_container.download_button(
                label="⬇ 下載藏密影像 (stego_output.png)",
                data=byte_im,
                file_name="stego_output.png",
                mime="image/png",
                use_container_width=True
            )
            st.success("✅ LSB 嵌入完成！")
        except Exception as e:
            st.error(str(e))

st.write("")

# 5. 卡方檢定 (Chi-square Attack)
st.markdown("#### ❺ 卡方檢定 (Chi-square Attack) 偵測結果 <span style='font-size:13px; color:#64748B; font-weight:normal;'>驗證目前嵌入是否容易被統計方法偵測</span>", unsafe_allow_html=True)

# 計算並繪製曲線
fig, ax = plt.subplots(figsize=(10, 3.2), dpi=100)
test_ratios = [0, 10, 25, 50, 75, 100]

if input_img:
    p_vals = []
    for r in test_ratios:
        if r == 0:
            p_vals.append(chi_square_attack(input_img))
        else:
            sim_img = embed_lsb(input_img, "SecretPayloadSampleData" * 30, r)
            p_vals.append(chi_square_attack(sim_img))
            
    ax.plot(test_ratios, [p_vals[0]]*len(test_ratios), linestyle="--", color="#3B82F6", label="原始影像")
    ax.plot(test_ratios, p_vals, color="#EA580C", marker="o", label=f"嵌入後影像 ({ratio}%)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("掃描比例 (%)")
    ax.set_ylabel("p-value")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)
else:
    ax.text(0.5, 0.5, "請先上傳影像以分析卡方檢定數據", ha='center', va='center', color="#94A3B8")
    ax.set_xticks([])
    ax.set_yticks([])

st.pyplot(fig)