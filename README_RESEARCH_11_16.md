# Khảo sát weighting11–16

Hướng dẫn hiện tại: [README](README.md); cơ chế, nguồn, tham số, giới hạn: [RESEARCH_METHODS](docs/RESEARCH_METHODS.md).

Campaign `weighting_research_20261028_214637_b9a2294e` đã chạy full:4 suite,100 predictor fit,60 selected test scores, seed 42/317/2026. LSR có auxiliary seed 42. Mọi suite có FROZEN; evaluation có SEALED/EVALUATION_COMPLETE. XGBoost attempt đầu thiếu package, sau repair có nhật ký và attempt hoàn tất. Không chạy repair sau SEALED.

Thứ tự campaign mới: **11 →12/13/14/15 →16**. 11 tạo cache train-only, lấy source theo `docs/research_sources.json`;12–15 chạy tuần tự;16 chỉ đánh giá sau khi mọi suite đã chọn khóa. Chọn INCLUDE_LSR trước khi tạo campaign. Child notebook có output ở `attempts/*/training_executed.ipynb`/`evaluation_executed.ipynb`.

Có nhiều campaign thì điền CAMPAIGN_DIR cụ thể. Source/config/data/environment phải khớp để resume. GitHub chỉ giữ phần evidence được chọn; cache/model/latent/full ZIP giữ local. Trên clone mới, tạo working copy riêng và campaign/cache mới, không sửa manifest cũ. Đọc bảng/chạy verify không cần train hoặc GPU.

`evaluation/` có bảng từng seed, mean/SD, paper/adaptation, bootstrap và score/nhãn BAF thật. `docs/validation/SYNTHETIC_*` là kiểm tra kỹ thuật riêng. 17 bổ sung metric;18/19 thêm đối chứng;21 ghép bảng cuối. Chưa chạy cải tiến đề xuất.
