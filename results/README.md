# Kết quả được chọn

Bảng cuối ở [advisor_combined/RESEARCH_TABLE.md](advisor_combined/RESEARCH_TABLE.md). `advisor_step1/` giữ bảng/audit60 score ban đầu; `research_campaigns/.../evaluation/` giữ score/nhãn, bảng paper và bootstrap; `advisor_runs/` giữ đối chứng bổ sung và hai run kiểm chứng nhỏ CPU/CUDA riêng biệt. `runs/`/`comparisons/` giữ bảng benchmark lịch sử.

Chạy `python -B scripts/verify_repository.py` từ root để kiểm chứng không train. Phạm vi và hash ở `docs/repository/EVIDENCE_SCOPE.json`. Worker notebook, verbose log, trial không chọn và full-proof index được giữ local; main giữ canonical notebook output, manifests, FROZEN, selected COMPLETE records, bảng và score.

Manifest/record lịch sử có thể nhắc tới model/validation scores/split chỉ giữ trên máy. Clone mới không đủ payload để resume; không sửa manifest để giả lập. Xem README gốc để tạo campaign mới.
