# DS317 — Đánh trọng số mẫu trên BAF

Đồ án xử lý mất cân bằng dữ liệu Bank Account Fraud: class/cost weighting, chọn trọng số bằng validation và chuyển giao các cơ chế weighting từ paper sang BAF.

**Kết quả:** [bảng tổng hợp](results/advisor_combined/RESEARCH_TABLE.md) · [phương pháp/protocol](docs/RESEARCH_METHODS.md) · [phạm vi kiểm chứng](docs/advisor/IMPLEMENTATION_SCOPE.md).

## Cấu trúc

| Thư mục | Vai trò |
|---|---|
| `notebooks/` | 20 notebook chính có output, từ01–09 và11–21 |
| `src/`, `supplement/` | Dữ liệu, mô hình, weighting và tổng hợp kết quả |
| `results/` | Bảng, cấu hình, lựa chọn đã khóa, score/nhãn để kiểm chứng |
| `docs/`, `references/` | Protocol, khảo sát, source lock, giấy phép và hướng dẫn chi tiết |
| `scripts/` | Verifier đọc lại kết quả và kiểm tra nhẹ notebook20 |

Dataset, model, cache, full proof và log/worker notebook lặp giữ local. Các báo cáo dọn repository, inventory và profile cài phụ đã gỡ khỏi main; lệnh cài tương ứng được ghi bên dưới. Giữ lịch sử Git và bản đầy đủ tại máy.

## Cài đặt

Môi trường đã chạy: Windows, Python **3.14.3**, PyTorch **2.14.1+cu130**, CUDA **13.0**, RTX3050 6GB. [Snapshot môi trường](docs/repository/ENVIRONMENT.json) và manifest từng run giữ phiên bản thực tế. Không tự cài lại Torch/CUDA hay cài chồng các thiết lập cũ.

Chỉ đọc/kiểm chứng kết quả, không cần GPU:

```powershell
py -m venv .venv
$Python = '.\.venv\Scripts\python.exe'
& $Python -m pip install -r requirements.txt
& $Python -m pip install nbformat ipython
& $Python -B scripts/verify_repository.py
```

Verifier cần Git và working copy có `.git`; nó đối chiếu dung lượng/SHA-256 toàn bộ file theo Git index, kiểm notebook, tính lại75 score BAF và30 bộ dự đoán nhỏ, đối chiếu bảng. Thêm `--imports` khi có môi trường huấn luyện. Script không train, tải dữ liệu, chạy HPO hay bootstrap; `--output` ghi ra ngoài project.

Để train11–21, chọn môi trường đã có Torch phù hợp rồi cài `requirements-research.txt`. Hai file requirements chính phục vụ baseline và nghiên cứu; không gộp chúng. Các gói phụ cho07/08/09 dưới đây giữ nguyên ràng buộc của profile cũ; chỉ cài nhóm thực sự cần, không chạy cả ba trên một môi trường đang khớp manifest:

```powershell
# Notebook07: bổ sung giao diện replay GRU; dùng môi trường ML/Torch sẵn có.
& $Python -m pip install "nbformat>=5.10" "ipython>=8.20" "tqdm>=4.66"

# Notebook08: tương thích Feedzai; legacy Optuna/Aequitas tải riêng bởi notebook.
& $Python -m pip install "lightgbm==4.7.0" "nbformat==5.11.1" "ipython==9.17.1" "PyYAML==6.0.3" "seaborn>=0.13.2"

# Notebook09: thiết lập đã dùng cho chọn class weight.
& $Python -m pip install "lightgbm==4.7.0" "nbformat==5.11.1" "ipython==9.17.1" "psutil==7.2.2" "matplotlib>=3.8"
```

Trên máy hiện tại dùng `$Python = 'D:\UIT\.venvs\ml-gpu\Scripts\python.exe'`. Nếu môi trường đã khớp manifest thì không cần cài lại. Cài theo khoảng phiên bản không bảo đảm tái lập bitwise trên máy khác.

## Dữ liệu

Tải **Base.csv** từ [BAF trên Kaggle](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) theo [nguồn Feedzai](https://github.com/feedzai/bank-account-fraud), đặt ở `data/raw/Base.csv`. Không commit dữ liệu/token. Bản đã dùng có1.000.000 dòng,11.029 fraud; SHA-256:

```text
7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809
```

Kiểm bằng `Get-FileHash data/raw/Base.csv -Algorithm SHA256`. Notebook20 tự tải MNIST/CIFAR-10 có kiểm MD5 vào `data/external/advisor`. Không cần các dataset để kiểm chứng score đã lưu.

## Thứ tự notebook

Mở từ project root hoặc `notebooks/`, chọn đúng kernel và đọc cấu hình trước Run All. Notebook huấn luyện có thể chạy nhiều giờ; khi chỉ xem báo cáo dùng output sẵn có hoặc verifier.

| Notebook | Nội dung |
|---|---|
|01 →02 →03 | Baseline LR/LightGBM → cost weighting → so sánh validation |
|04 | GRU pilot; không dùng làm benchmark full |
|05 →06 | GRU full seed1234 → thêm317/2026; dùng source lịch sử |
|07 | Replay GRU nguồn: HPO10 và fixed recipe |
|08 | Feedzai Base100 trial; top-five test chỉ để mô tả |
|09 | Chọn none/balanced/tuned trên validation6 rồi đánh giá tháng7 |
|11 →12/13/14/15 →16 | Chuẩn bị → losses/meta/LSR/XGBoost → bảng paper/bootstrap |
|17 | Audit60 score và tổng hợp độ đo |
|18,19 →21 | CE batch256/sampling baselines → bảng cuối, đã chạy xong |
|20 | Kiểm công thức và chạy nhỏ CB/LRE/MWN trên MNIST/CIFAR-10 |

**05/06:** source hiện tại sẽ bị guard lịch sử từ chối. Snapshot đúng nằm trong `results/runs/gru_author_fixed_full_20261027_123100_deeb80e4/evidence/src/` và `results/runs/gru_author_fixed_full_extra_seeds_20261027_182144_4cb8cfb7/evidence/src/`. Tái chạy cần working copy riêng, môi trường cũ và full base run local; không bỏ guard hay ghi đè source hiện tại.

**20:** `RESUME_RUN = None` chọn theo config/source/environment hiện tại. Explicit resume phải khớp toàn bộ identity và vị trí canonical của `reporting.launch`. Output cũ được giữ; thay đổi selector được kiểm tra nhẹ, không train lại.

**Campaign mới:** GitHub không có cache/checkpoint để resume run cũ. Dùng working copy riêng, chạy11 tạo cache/campaign từ Base.csv, chọn CAMPAIGN_DIR cụ thể trong12–16/18/19 nếu có nhiều campaign. Chạy suite tuần tự;16 cần mọi suite đã chọn có FROZEN. Không sửa manifest lịch sử để giả lập resume.

## Nơi xem kết quả

- `results/advisor_combined/`: bảng cuối27 nhóm/81 dòng seed, gồm75 score duy nhất và6 dòng meta tái dùng.
- `results/advisor_step1/`: bảng/audit60 score nghiên cứu gốc.
- `results/research_campaigns/weighting_research_20261028_214637_b9a2294e/`: config, FROZEN, validation summary, selected records; `evaluation/` giữ score/nhãn, bảng paper và bootstrap đã chạy.
- `results/advisor_runs/`: đối chứng CE/sampling và hai run SmallCNN CPU/CUDA riêng biệt,15 bộ dự đoán mỗi run.
- `results/runs/`, `results/comparisons/`: benchmark lịch sử và snapshot source cần thiết.

Chỉ so weighting trong cùng backbone/protocol. Tháng7 đã xem khi phát triển, nên là development test; SD seed không thay CI test. Bootstrap gốc không tự bao phủ18/19. Image-paper là BAF adaptation, LSR có budget giới hạn/chung auxiliary seed42, SmallCNN20 không tái lập đầy đủ paper. Chưa chạy CCRW-E/P hoặc cải tiến đề xuất. Xem [nguồn và giấy phép](THIRD_PARTY_NOTICES.md).
