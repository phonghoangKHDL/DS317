# DS317 — Xử lý mất cân bằng dữ liệu BAF bằng đánh trọng số mẫu

Đồ án khảo sát class/cost weighting, chọn trọng số trên validation và chuyển giao các cơ chế weighting từ paper sang Bank Account Fraud (BAF). Repo giữ mã nguồn, notebook có output, bảng khảo sát, kết quả và phần minh chứng có thể tính lại. CCRW-E/P là hướng dự kiến, **chưa có thực nghiệm** trong bản này.

Đọc [bảng kết quả cuối](results/advisor_combined/RESEARCH_TABLE.md), [phạm vi phương pháp](docs/RESEARCH_METHODS.md) và [báo cáo rà soát](REPOSITORY_REVIEW.md). [GITHUB_FILES.txt](docs/repository/GITHUB_FILES.txt) là danh sách file chính thức; [FILE_INVENTORY.csv](FILE_INVENTORY.csv) ghi từng quyết định giữ/lưu máy/lưu trữ.

## Cấu trúc

```text
src/                 baf.py: dữ liệu, metric, LR/LightGBM, chọn class weight
                     gru.py: GRU, loss/meta weighting, LSR, XGBoost, campaign
supplement/          reporting.py và training.py cho notebook 17–21
notebooks/           20 notebook chính: 01–09 và 11–21; không có bước 10
docs/                protocol, nguồn khóa commit/hash, synthetic checks, giấy phép
references/          khảo sát ban đầu (.md/.xlsx/.csv), notebook GRU nguồn
scripts/             helper cài đặt và verify_repository.py chỉ kiểm chứng
results/             kết quả được chọn, giữ nguyên đường dẫn và byte
data/raw/README.md   hướng dẫn lấy Base.csv; dữ liệu thật giữ tại máy
```

`external/`, cache/latent, model, dữ liệu và ZIP proof không đưa vào Git. Chúng vẫn ở vị trí cũ trên máy để tránh phá resume. Manifest lịch sử có thể chứa đường dẫn tuyệt đối của máy đã chạy: đó là provenance, không phải cấu hình portable để resume trên máy khác. Phần minh chứng công khai được liệt kê trong [EVIDENCE_SCOPE.json](docs/repository/EVIDENCE_SCOPE.json).

## Môi trường và cài đặt

Đã kiểm tra ngày 06/11/2026: Windows, Python 3.14.3, PyTorch **2.14.1+cu130**, CUDA build 13.0, RTX 3050 6GB; `pip check` đạt. Phiên bản đầy đủ và kiểm tra từng profile nằm ở [ENVIRONMENT.json](docs/repository/ENVIRONMENT.json). Môi trường CPU/phiên bản cũ được giữ riêng trong manifest từng run.

Trên máy hiện tại, dùng đúng interpreter/kernel:

```powershell
$Python = 'D:\UIT\.venvs\ml-gpu\Scripts\python.exe'
& $Python -m pip check
& $Python -B scripts/verify_repository.py --imports
```

Trên máy mới, tạo môi trường riêng để đọc/kiểm chứng, không cần GPU:

```powershell
py -m venv .venv
$Python = '.\.venv\Scripts\python.exe'
& $Python -m pip install -r requirements.txt
& $Python -m pip install nbformat ipython
& $Python -B scripts/verify_repository.py
```

Script verify đối chiếu byte size và SHA-256 của **mọi file chính thức** với FILE_INVENTORY.csv, kiểm notebook/đường dẫn và tính lại 75 bộ score BAF, 30 fit dữ liệu gốc (15 CPU + 15 CUDA). Inventory tự kiểm dung lượng nhưng để trống SHA-256 của chính nó để tránh tự tham chiếu; mọi file khác bắt buộc có hash. Script cũng kiểm byte/hash lịch sử của paper_survey.csv và chạy các fixture nhẹ cho chọn run notebook20/inventory. Không train, tải dữ liệu hay chạy lại bootstrap.

Nếu dùng `--output`, ghi báo cáo **bên ngoài working copy đang kiểm tra** để không thay đổi một file đã được inventory bảo vệ. `docs/repository/VERIFICATION.json` là báo cáo đã đóng gói và có hash trong inventory.

Để train trên máy mới, cài PyTorch/CUDA phù hợp trước rồi chọn profile bên dưới. Để tái lập môi trường đã báo cáo, đối chiếu phiên bản trong manifest/ENVIRONMENT.json; cài theo khoảng phiên bản **không bảo đảm** tái lập bitwise. Repo không tự cài hoặc thay Torch hiện có.

| Profile giữ nguyên | Vai trò |
|---|---|
| `requirements.txt` | LR/LightGBM, tiền xử lý và kernel cho 01–03; không cài Torch |
| `requirements-author07.txt` | nbformat/IPython/tqdm cho replay GRU; cần thêm môi trường ML/Torch |
| `requirements-feedzai08.txt` | Profile riêng cho 08; notebook cài Optuna/Aequitas vào `external/feedzai08_runtime/packages` |
| `requirements-step09.txt` | Profile riêng cho 09, dùng môi trường CUDA hiện có |
| `requirements-research.txt` | 11–21; có XGBoost, không yêu cầu Torch |

Không cài chồng tất cả profile. Nếu môi trường khớp manifest và import được thì không cần cài lại. Hai helper `install_research.ps1 -PythonExe <python>` và `scripts/Install-BAFAdvisorDeps.ps1 -Python <python>` dành cho cài bổ sung có chủ đích; chúng có mặc định đường dẫn ổ D và pip có thể đổi phiên bản gói nhẹ. Đọc script trước khi dùng trên máy khác. Không cần chạy helper khi chỉ xem kết quả.

## Lấy dữ liệu

Theo [nguồn Feedzai](https://github.com/feedzai/bank-account-fraud), tải BAF từ [Kaggle](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022), lấy **Base.csv** đầy đủ và đặt ở `data/raw/Base.csv`. Không commit tài khoản/token Kaggle hay dữ liệu. Đã dùng 1.000.000 dòng, 11.029 fraud, tháng0–7, SHA-256:

```text
7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809
```

PowerShell: `Get-FileHash data/raw/Base.csv -Algorithm SHA256`. Không ghi lại CSV rồi bỏ qua hash. Notebook20 tải MNIST/CIFAR-10 và kiểm MD5 vào `data/external/advisor`; có thể chọn từng dataset trong cấu hình. Không cần các dataset để verify score đã lưu.

Git cần có trong PATH để 07/08/11 lấy source đúng commit. Không cần `git submodule update`: nguồn tải về là thư mục local được ignore.

## Notebook và thứ tự chạy

Mở từ project root hoặc `notebooks/`, chọn đúng kernel, đọc cell cấu hình trước Run All. **Notebook huấn luyện có thể chạy nhiều giờ**; để xem báo cáo, đọc output sẵn có hoặc chạy script verify.

| Notebook | Vai trò |
|---|---|
| 01 → 02 → 03 | Baseline LR/LightGBM, cost weighting, rồi so sánh; 03 không fit |
| 04 | Pilot GRU một epoch trên mẫu nhỏ; không đưa vào benchmark full |
| 05 → 06 | GRU full seed 1234 rồi thêm 317/2026; thực nghiệm lịch sử có source lock, xem lưu ý bên dưới |
| 07 | Replay GRU nguồn: HPO10 + fixed recipe; khác 05/06 không chạy HPO |
| 08 | Feedzai Base 100 trial; train0–5/test6–7; chọn top-five trên test chỉ để mô tả |
| 09 | Chọn none/balanced/tuned trên validation6, khóa ngưỡng rồi đánh giá tháng7 |
| 11 → 12/13/14/15 → 16 | Tạo campaign/cache/source lock → losses/meta/LSR/XGBoost → bảng paper/bootstrap |
| 17 | Audit 60 score và tổng hợp độ đo đầy đủ; không fit |
| 18, 19 → 21 | CE batch256 và sampling baselines trên cùng campaign → ghép bảng; đều đã chạy xong |
| 20 | Kiểm chứng công thức và chạy nhỏ CB/LRE/MWN trên MNIST/CIFAR-10; bản hiện tại có retry ghi CSV |

Không có notebook chính nào trùng hoàn toàn source. 17 bổ sung metric cho 16; 21 mở rộng bảng 17, tái dùng LRE/MWN chứ không fit mới hai phương pháp đó. Notebook `docs/validation/SYNTHETIC_*` chỉ là kiểm tra kỹ thuật.

**GRU lịch sử 05/06:** 05 dừng với source hiện tại vì SHA cũ; 06 dừng ở training signature so với base run. Không bỏ các kiểm tra đó. Snapshot đúng ở `results/runs/gru_author_fixed_full_20261027_123100_deeb80e4/evidence/src/` (05) và `results/runs/gru_author_fixed_full_extra_seeds_20261027_182144_4cb8cfb7/evidence/src/` (06). Để chạy lại lịch sử, dùng bản sao dự án riêng với source tương ứng và môi trường cũ. 06 cần đầy đủ base run local, gồm predictions/split/model; khi tạo run mới phải cập nhật `BASE_RUN_DIR`. Các payload này không nằm trong bản GitHub gọn. Không chép đè `src/` của campaign hiện tại.

**Notebook20:** `RESUME_RUN = None` suy ra run từ cấu hình, hash source và môi trường hiện tại; không tự ưu tiên run cũ trong log. Muốn resume một run cũ, điền đường dẫn rõ ràng; toàn bộ manifest identity phải khớp. Đường dẫn phải là vị trí canonical theo identity như `reporting.launch` đang dùng, nên run đã di chuyển bị từ chối trước khi launch. Output đang lưu là minh chứng lần chạy cũ; sửa chọn run chỉ được kiểm tra nhẹ, không chạy lại train.

**Campaign mới:** GitHub giữ manifest/score nhưng không giữ cache/checkpoint để resume. Dùng working copy riêng để train, không lấy các run minh chứng làm cache. Chạy11 tạo campaign/cache mới từ Base.csv, điền `CAMPAIGN_DIR` mới trong 12–16/18/19 nếu có nhiều campaign. Chạy suite tuần tự; 16 cần mọi suite đã chọn có FROZEN. Không sửa manifest cũ để giả lập resume. Trên máy gốc vẫn dùng cache cũ được khi source/config/environment khớp.

## Nơi xem kết quả và giới hạn

- `results/advisor_combined/RESEARCH_TABLE.md`: bảng cuối, 27 nhóm method/backbone, 81 dòng seed; **75 score BAF duy nhất** và 6 dòng LRE/MWN tái dùng cho đối chiếu CE batch256.
- `results/advisor_step1/`: 60 score/20 phương pháp gốc, bảng metric và audit/provenance.
- `results/research_campaigns/weighting_research_20261028_214637_b9a2294e/evaluation/`: score/nhãn, bảng từng seed, `research_table.csv`, `paper_research_table.csv`, `paired_bootstrap.csv`.
- `results/advisor_runs/ce_0c139aa5b451/`, `sampling_1827c85af012/`: 5 + 20 predictor fit, 3 + 12 selected test scores.
- `results/advisor_runs/original_checks_582e4a8ada1a/`: 15 fit SmallCNN CUDA; `original_checks_3848ef4489af` là 15 fit CPU bàn giao, giữ riêng.
- `results/runs/`, `results/comparisons/`: kết quả lịch sử01–09. ZIP proof đầy đủ chỉ giữ local.

Chỉ so weighting trong cùng backbone/feature/protocol. Image-paper chuyển sang BAF không phải tái lập toàn bộ paper. LSR giới hạn ngân sách, dùng chung auxiliary seed 42. Tháng7 đã xem khi phát triển: đây là development test. SD seed không thay CI lấy mẫu test. Bootstrap 1.000 lần có sẵn thuộc11–16, không tự áp dụng cho18/19. Đọc [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) trước khi dùng lại mã bên thứ ba.
