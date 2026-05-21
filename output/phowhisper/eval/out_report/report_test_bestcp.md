# Báo Cáo Chuyên Sâu Cấp Hệ Thống (Diagnostic ASR - Whisper)

> **Nguyên tắc đánh giá:** Báo cáo này áp dụng quy chuẩn phân tách thành phần lỗi (Diagnostic Decomposition) giống hệt với tư duy chấm điểm của thư viện Diarization (Pyannote). Thay vì chỉ in 1 điểm số tù mù, hệ thống bóc tách lỗi thành I-D-S và chia vùng theo từng Datasets.

## 1. Kết Quả Tổng Quan (Global Metrics)
- **Word Error Rate (WER TỔNG):** `14.17%`
  - Lỗi Nhận dạng sai (Substitutions - S): `5.89%` *(Đọc A viết thành B)*
  - Lỗi Nuốt chữ/Mất tiếng (Deletions - D): `4.20%` *(Nói mà máy không hiện chữ)*
  - Lỗi Bịa thêm chữ (Insertions - I): `4.08%` *(Máy ngáo tự sinh thêm chữ)*

## 2. Phân nhỏ Tỷ lệ lỗi theo Tập Dữ Liệu (Break-down by Domain/Group)

| Tên Dataset (Group) | Tổng độ dài từ | WER Tổng (%) | (S) Sai (%) | (D) Nuốt (%) | (I) Bịa (%) |
|---------------------|----------------|--------------|-------------|--------------|-------------|
| chuyen_ho_chuyen_minh | 9872 từ | **17.93%** | 6.75 | 5.51 | 5.67 |
| coi_mo | 8911 từ | **23.77%** | 10.89 | 6.39 | 6.50 |
| conan | 3550 từ | **7.58%** | 4.00 | 2.65 | 0.93 |
| dustin_on_go | 13291 từ | **14.58%** | 5.94 | 4.93 | 3.71 |
| vif | 13597 từ | **6.48%** | 2.43 | 1.51 | 2.54 |

## 3. Khám nghiệm tử thi dữ liệu (Forensics & Data Diff)
Để phân tích nguyên nhân tại sao mô hình làm tệ ở một tập dữ liệu cụ thể, chúng tôi đã trích xuất sẵn đối chiếu văn bản 1-1 (Ground Truth đấu với Model) cho toàn bộ 1655 test cases vào tệp Excel/CSV sau:
👉 `predictions_versus_groundtruth.csv`
