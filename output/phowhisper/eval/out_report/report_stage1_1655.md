# Báo Cáo Chuyên Sâu Cấp Hệ Thống (Diagnostic ASR - Whisper)

> **Nguyên tắc đánh giá:** Báo cáo này áp dụng quy chuẩn phân tách thành phần lỗi (Diagnostic Decomposition) giống hệt với tư duy chấm điểm của thư viện Diarization (Pyannote). Thay vì chỉ in 1 điểm số tù mù, hệ thống bóc tách lỗi thành I-D-S và chia vùng theo từng Datasets.

## 1. Kết Quả Tổng Quan (Global Metrics)
- **Word Error Rate (WER TỔNG):** `13.63%`
  - Lỗi Nhận dạng sai (Substitutions - S): `5.58%` *(Đọc A viết thành B)*
  - Lỗi Nuốt chữ/Mất tiếng (Deletions - D): `4.46%` *(Nói mà máy không hiện chữ)*
  - Lỗi Bịa thêm chữ (Insertions - I): `3.60%` *(Máy ngáo tự sinh thêm chữ)*

## 2. Phân nhỏ Tỷ lệ lỗi theo Tập Dữ Liệu (Break-down by Domain/Group)

| Tên Dataset (Group) | Tổng độ dài từ | WER Tổng (%) | (S) Sai (%) | (D) Nuốt (%) | (I) Bịa (%) |
|---------------------|----------------|--------------|-------------|--------------|-------------|
| chuyen_ho_chuyen_minh | 9872 từ | **17.56%** | 6.61 | 5.95 | 5.00 |
| coi_mo | 8911 từ | **19.76%** | 9.62 | 6.91 | 3.23 |
| conan | 3550 từ | **7.46%** | 3.86 | 2.85 | 0.76 |
| dustin_on_go | 13291 từ | **15.88%** | 5.86 | 5.21 | 4.80 |
| vif | 13597 từ | **6.18%** | 2.35 | 1.45 | 2.38 |

## 3. Khám nghiệm tử thi dữ liệu (Forensics & Data Diff)
Để phân tích nguyên nhân tại sao mô hình làm tệ ở một tập dữ liệu cụ thể, chúng tôi đã trích xuất sẵn đối chiếu văn bản 1-1 (Ground Truth đấu với Model) cho toàn bộ 1655 test cases vào tệp Excel/CSV sau:
👉 `predictions_versus_groundtruth.csv`
