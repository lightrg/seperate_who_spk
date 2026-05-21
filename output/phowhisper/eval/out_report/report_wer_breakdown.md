# Báo Cáo Chuyên Sâu Cấp Hệ Thống (Diagnostic ASR - Whisper)

> **Nguyên tắc đánh giá:** Báo cáo này áp dụng quy chuẩn phân tách thành phần lỗi (Diagnostic Decomposition) giống hệt với tư duy chấm điểm của thư viện Diarization (Pyannote). Thay vì chỉ in 1 điểm số tù mù, hệ thống bóc tách lỗi thành I-D-S và chia vùng theo từng Datasets.

## 1. Kết Quả Tổng Quan (Global Metrics)
- **Word Error Rate (WER TỔNG):** `11.84%`
  - Lỗi Nhận dạng sai (Substitutions - S): `5.85%` *(Đọc A viết thành B)*
  - Lỗi Nuốt chữ/Mất tiếng (Deletions - D): `2.45%` *(Nói mà máy không hiện chữ)*
  - Lỗi Bịa thêm chữ (Insertions - I): `3.54%` *(Máy ngáo tự sinh thêm chữ)*

## 2. Phân nhỏ Tỷ lệ lỗi theo Tập Dữ Liệu (Break-down by Domain/Group)

| Tên Dataset (Group) | Tổng độ dài từ | WER Tổng (%) | (S) Sai (%) | (D) Nuốt (%) | (I) Bịa (%) |
|---------------------|----------------|--------------|-------------|--------------|-------------|
| chuyen_ho_chuyen_minh | 9872 từ | **14.01%** | 6.56 | 3.27 | 4.17 |
| coi_mo | 8911 từ | **19.67%** | 11.13 | 3.22 | 5.32 |
| conan | 3550 từ | **5.44%** | 3.86 | 0.82 | 0.76 |
| dustin_on_go | 13291 từ | **12.24%** | 5.79 | 2.79 | 3.66 |
| vif | 13597 từ | **6.41%** | 2.43 | 1.44 | 2.54 |

## 3. Khám nghiệm tử thi dữ liệu (Forensics & Data Diff)
Để phân tích nguyên nhân tại sao mô hình làm tệ ở một tập dữ liệu cụ thể, chúng tôi đã trích xuất sẵn đối chiếu văn bản 1-1 (Ground Truth đấu với Model) cho toàn bộ 1554 test cases vào tệp Excel/CSV sau:
👉 `predictions_versus_groundtruth.csv`
