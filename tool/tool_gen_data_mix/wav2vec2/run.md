# Quick Commands

## Wav2Vec2 Alignment (Recommended - No conda needed!)
```bash
#chuyển audio wav + promt sang dạng word time stamp

# 1. Install dependencies
pip install -r requirements_wav2vec2.txt

# 2. Run alignment (auto-downloads model on first run)
python align_wav2vec2.py

# 3. Validate
python validate_alignment.py --input-dir data/vivos/train/aligned_wav2vec2
``
chuyển sang dạng work