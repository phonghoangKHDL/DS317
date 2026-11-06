# DS317 — Đánh trọng số mẫu trên BAF

Đồ án xử lý mất cân bằng dữ liệu Bank Account Fraud (BAF): khảo sát class/cost weighting, chọn trọng số bằng validation và chuyển giao các cơ chế từ paper sang dữ liệu bảng.

**Xem kết quả:** [bảng tổng hợp](results/advisor_combined/RESEARCH_TABLE.md), [protocol/phương pháp](docs/RESEARCH_METHODS.md), [phạm vi kiểm chứng](docs/advisor/IMPLEMENTATION_SCOPE.md).

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `src/` | Dữ liệu, metric, LR/LightGBM, GRU và các adapter nghiên cứu |
| `supplement/` | Tổng hợp metric và đối chứng bổ sung |
| `notebooks/` | 20  notebook chính có output; không có bước10 |
| `results/` | Bảng, score/nhãn, cấu hình và lựa chọn đã khóa để kiểm chứng |
| `references/`, `docs/` | Khảo sát, protocol, source lock, giấy phép và hướng dẫn từng bước |
| `scripts/` | Kiểm chứng không train, kiểm tra nhẹ và helper cài đặt |

Các notebook con, log chi tiết, trial không được chọn, model, cache và ZIP proof đầy đủ giữ local. Repo chỉ giữ phần minh chứng cần để đọc báo cáo và tính lại kết quả. Xem [REPOSITORY_REVIEW.md](REPOSITORY_REVIEW.md), [inventory](FILE_INVENTORY.csv) và [danh sách file](docs/repository/GITHUB_FILES.txt).

## Môi trường

Máy đã chạy: Windows, Python 3 .14 .3 , PyTorch 2 .14 .1 +cu130/CUDA13.0 , RTX 3050  6GB. Snapshot môi trường ở [ENVIRONMENT.json](docs/repository/ENVIRONMENT.json); manifest từng run mô tả môi trường của chính run đó. Không gộp các profile dependencies hoặc tự thay bản Torch/CUDA.

Đọc và kiểm chứng trên môi trường mới không cần GPU:

```powershell
py -m venv .venv
$Python = '.\.venv\Scripts\python.exe'
& $Python -m pip install -r requirements.txt
& $Python -m pip install nbformat ipython
& $Python -B scripts/verify_repository.py
```

Trên máy hiện tại dùng `$Python = 'D:\UIT\.venvs\ml-gpu\Scripts\python.exe'` và thêm `--imports` để kiểm tra import huấn luyện. Verifier đối chiếu dung lượng/SHA-256  mọi file chính thức, kiểm notebook 20 , tính lại **75  score BAF** và **30  bộ dự đoán kiểm chứng nhỏ**, kiểm bảng tổng hợp; không train, chạy HPO hay bootstrap. Inventory tự kiểm dung lượng và để trống hash của chính nó; mọi file khác bắt buộc có hash. Nếu dùng `--output`, ghi ngoài working copy để không làm thay đổi file đã kiểm.

| Profile | Dùng cho |
|---|---|
| `requirements.txt` | Baseline, tiền xử lý và kernel |
| `requirements-author07.txt` | Bổ sung cho replay GRU07 |
| `requirements-feedzai08.txt` | Feedzai08; legacy packages tải riêng vào `external/` |
| `requirements-step09.txt` | Chọn class weight09 |
| `requirements-research.txt` | Nghiên cứu11–21 ; không cài Torch |

Để train trên máy khác, cài Torch/CUDA phù hợp trước rồi chọn đúng profile, không cài chồng cả năm. Các khoảng phiên bản không bảo đảm tái lập bitwise; đối chiếu manifest nếu cần đúng môi trường cũ. Helper `install_research.ps1 -PythonExe <python>` và `scripts/Install-BAFAdvisorDeps.ps1 -Python <python>` có mặc định ổ D; không cần chạy chúng chỉ để xem kết quả.

## Dữ liệu

Tải **Base.csv** từ [BAF trên Kaggle](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022 ) theo [nguồn Feedzai](https://github.com/feedzai/bank-account-fraud), đặt vào `data/raw/Base.csv`. Không commit dữ liệu hoặc token Kaggle.

Đã dùng1.000 .000  dòng,11 .029  fraud; SHA-256 :

```text
7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809
```

Kiểm bằng `Get-FileHash data/raw/Base.csv -Algorithm SHA256`. Notebook20 tải MNIST/CIFAR-10  có kiểm MD5 vào `data/external/advisor`. Không cần tải dữ liệu để verify score đã lưu. Git cần có trong PATH để07/08/11 tải source đúng commit; không có submodule.

## Notebook và thứ tự chạy

Mở từ project root hoặc `notebooks/`, chọn đúng kernel và đọc cấu hình trước Run All. Notebook huấn luyện có thể chạy nhiều giờ; khi chỉ xem báo cáo hãy dùng output sẵn có hoặc verifier.

| Notebook | Vai trò |
|---|---|
|01  →02  →03  | Baseline LR/LightGBM → cost weighting → so sánh validation |
|04  | GRU pilot; không tính vào benchmark full |
|05  →06  | GRU full seed 1234  → thêm317/2026; source lịch sử, xem lưu ý bên dưới |
|07  | Replay GRU nguồn: HPO10 và fixed recipe |
|08  | Feedzai Base 100  trial; top-five test chỉ để mô tả |
|09  | Chọn none/balanced/tuned trên validation6 rồi đánh giá tháng7 |
|11  →12 /13/14/15 →16  | Tạo campaign → losses/meta/LSR/XGBoost → bảng paper/bootstrap |
|17  | Audit60 score và tổng hợp độ đo đầy đủ |
|18 ,19  →21  | CE batch256 và sampling baselines → bảng cuối; đã chạy xong |
|20  | Kiểm tra công thức và chạy nhỏ CB/LRE/MWN trên MNIST/CIFAR-10  |

**05 /06:** guard source sẽ từ chối bản mở rộng hiện tại. Snapshot đúng ở `results/runs/gru_author_fixed_full_20261027_123100_deeb80e4/evidence/src/` và `results/runs/gru_author_fixed_full_extra_seeds_20261027_182144_4cb8cfb7/evidence/src/`. Để train lại lịch sử dùng working copy riêng, source/môi trường đúng phiên bản;06  cần full base run local và BASE_RUN_DIR đúng. Không bỏ guard hoặc ghi đè source hiện tại.

**20 :** `RESUME_RUN = None` chọn theo config/source/environment hiện tại. Explicit resume phải khớp toàn bộ identity và vị trí canonical của `reporting.launch`; run đã di chuyển bị từ chối trước launch. Output được giữ từ lần chạy cũ, sửa selector chỉ được kiểm tra nhẹ.

**Campaign mới:** GitHub không chứa cache/checkpoint để resume run cũ. Tạo working copy riêng, chạy11 với Base.csv để tạo cache/campaign mới, chọn CAMPAIGN_DIR cụ thể trong12–16 /18/19 nếu có nhiều campaign. Chạy các suite tuần tự;16  cần mọi suite đã chọn có FROZEN. Không sửa manifest lịch sử để giả lập resume.

## Kết quả và giới hạn

- `results/advisor_combined/`: bảng cuối27 nhóm/81 dòng seed, gồm75 score duy nhất và6 dòng LRE/MWN tái dùng.
- `results/advisor_step1/`: bảng/audit của60 score nghiên cứu gốc.
- `results/research_campaigns/weighting_research_20261028_214637_b9a2294e/`: cấu hình, lựa chọn khóa, validation summary, selected records; `evaluation/` có60 score/nhãn, bảng paper và bootstrap đã chạy.
- `results/advisor_runs/`: CE/sampling thêm15 score; hai run SmallCNN CPU/CUDA giữ riêng,15  bộ dự đoán mỗi run.
- `results/runs/`, `results/comparisons/`: bảng benchmark lịch sử, cấu hình và source snapshot cần thiết.

Chỉ so weighting trong cùng backbone/protocol. Tháng7 đã xem trong phát triển: đây là development test. SD seed không thay CI test; bootstrap gốc không tự áp dụng cho 18 /19. Các image-paper là BAF adaptation, LSR có ngân sách giới hạn/chung auxiliary seed 42 ; SmallCNN20 không tái lập đầy đủ paper. Chưa chạy CCRW-E/P hay cải tiến đề xuất. Xem [nguồn và giấy phép](THIRD_PARTY_NOTICES.md).
