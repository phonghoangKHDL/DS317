# Kết quả đã chạy và minh chứng GitHub

Bảng cuối: [advisor_combined/RESEARCH_TABLE.md](advisor_combined/RESEARCH_TABLE.md). `advisor_step1/` giữ60 score nghiên cứu; `advisor_runs/` giữ CE/sampling và kiểm chứng nhỏ CPU/CUDA; `research_campaigns/.../evaluation/` giữ60 score/nhãn để tính metric. `runs/`/`comparisons/` giữ bảng và provenance lịch sử. `checks/` là kiểm tra kỹ thuật, không phải benchmark BAF mới.

Chạy `python -B scripts/verify_repository.py` từ root, không train. Phạm vi/hash ở `docs/repository/EVIDENCE_SCOPE.json`. Các index `sha256.json`, `COMPLETE.json`, `FROZEN.json` lịch sử giữ nguyên; chúng có thể nhắc tới model/validation scores/split chỉ còn ở máy. Bản GitHub không phải full proof có thể resume.

Dữ liệu/cache/checkpoint/latent/raw predictions một số run cũ và toàn bộ `proofs/` giữ tại vị trí cũ trên máy, không stage. Attempt lỗi giữ làm lịch sử trước attempt thành công, không cộng thành kết quả mới. `original_checks_3848ef4489af` là CPU; `original_checks_582e4a8ada1a` là CUDA; không trộn kết quả.
