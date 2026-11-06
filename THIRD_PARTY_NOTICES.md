# Nguồn và giấy phép bên thứ ba

Giữ attribution theo từng nguồn khóa, không áp một giấy phép chung lên toàn repo. Nhóm chưa khai báo giấy phép riêng cho mã/tài liệu tự viết. LICENSE/NOTICE gốc giữ nguyên byte; tác giả/nguồn ở README và manifest liên quan.

| Nguồn | Lock | Giấy phép / phần sử dụng |
|---|---|---|
| wn-nwN/fraud-detection-project-Nov2025 | `6e8252cb9080378905b2aae30033c5946b72335b`, `references/gru_author/source_manifest.json` | [Apache-2.0 gốc](references/gru_author/LICENSE); notebook tham chiếu và GRU chuyển triển khai |
| feedzai/bank-account-fraud | `3377cfd8fc20edcda3dd9dd14c10ed3dd276c6de`, notebook 08 | [LICENSE gốc](docs/third_party/feedzai/LICENSE), giữ thông báo sử dụng thương mại; wrapper/khác biệt có diff trong run evidence |
| richardaecn/class-balanced-loss | `docs/research_sources.json`, cb | [LICENSE](docs/advisor/upstream/cb/LICENSE), [README](docs/advisor/upstream/cb/README.md); reference effective-number/CB và port PyTorch |
| kaidic/LDAM-DRW | `docs/research_sources.json`, ldam | [LICENSE](docs/third_party/ldam/LICENSE); margin/DRW và normalized head |
| uber-research/learning-to-reweight-examples | `docs/research_sources.json`, lre | [LICENSE](docs/advisor/upstream/lre/LICENSE), [NOTICE](docs/advisor/upstream/lre/NOTICE); reference/meta-gradient port |
| xjtushujun/Meta-weight-net_class-imbalance | `docs/research_sources.json`, mwn | [LICENSE](docs/advisor/upstream/mwn/LICENSE); resnet.py dùng kiểm tra số học, trainer chỉ tham chiếu |
| jhwjhw0123/Imbalance-XGBoost | `docs/research_sources.json`, imxgb | [LICENSE](docs/third_party/imxgb/LICENSE); tải source loss đúng hash, wrapper API hiện tại |
| YunzeTong/latent-score-based-reweighting | `docs/research_sources.json`, lsr | Không thấy LICENSE trong checkout đã kiểm tra; không phân phối lại checkout/snapshot LSR ở GitHub. Giữ URL/commit/hash, kết quả và adapter của đồ án. Cần kiểm tra quyền nguồn trước khi phân phối lại mã tác giả. |

Repo đầy đủ tải lúc chạy11, không phải submodule. `docs/advisor/upstream/` là reference CB/LRE/MWN khóa theo `source_lock.json`. Trainer MWN có `cuda(async=True)` không parse được trên Python hiện tại; giữ nguyên để đối chiếu, không import trainer. Không sửa reference để che lỗi cú pháp lịch sử.

BAF/MNIST/CIFAR-10 không đóng kèm; lấy nguồn trong README/code và theo điều khoản từng dataset. Khảo sát paper/workbook giữ ở `references/survey_20261026/`. Kết quả BAF có adaptation, không thay cho kết quả paper gốc.
