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

Dataset, model, cache, full proof và log/worker notebook lặp giữ local. Các báo cáo quản trị và profile cài trùng đã gỡ khỏi main; người clone dùng requirements.txt và README này. Giữ lịch sử Git và bản đầy đủ tại máy.

## Clone và cài đặt

Người clone mới dùng **một file `requirements.txt`** cho luồng chính. File này chứa các gói dữ liệu, weighting, XGBoost, notebook/plot và PyTorch; không còn phải chọn giữa requirements thường và research.

Cần Git và Python3.11 trở lên. Ví dụ dưới đây dùng Python3.11 trên Windows; Linux/macOS dùng `python3.11 -m venv .venv` và `.venv/bin/python` thay đường dẫn interpreter.

```powershell
git clone https://github.com/phonghoangKHDL/DS317.git
cd DS317
py -3.11 -m venv .venv
$Python = '.\.venv\Scripts\python.exe'
& $Python -m pip install --upgrade pip
```

Chọn **một** cách cài PyTorch trong chính môi trường mới:

- **CPU:** dùng lệnh dưới. Đủ để kiểm chứng kết quả và import/chạy các phương pháp hỗ trợ CPU; training full có thể rất lâu. Notebook07 mặc định yêu cầu CUDA, notebook06 còn khóa môi trường cũ.
- **NVIDIA GPU:** lấy lệnh Windows/Linux, Pip, Python, CUDA phù hợp máy từ [bộ chọn chính thức của PyTorch](https://pytorch.org/get-started/locally/). Dùng `$Python -m pip` thay `pip3` trong lệnh đó. Không chọn CUDA theo máy tác giả nếu driver/máy của bạn khác.

```powershell
# Chỉ chạy dòng này khi chọn CPU.
& $Python -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu

# Sau khi đã chọn CPU hoặc CUDA, cài phần còn lại bằng cùng một file.
& $Python -m pip install -r requirements.txt
& $Python -m pip check
& $Python -B scripts/verify_repository.py --imports
```

`requirements.txt` chấp nhận Torch2.x đã cài, gồm bản CPU/CUDA; không dùng `--upgrade` khi muốn giữ phiên bản đó. Các bước trên dành cho **venv mới**. Môi trường `ml-gpu` cũ có manifest/source lock riêng, không cần cài lại. Máy tác giả đã dùng Python3.14.3, Torch2.14.1+cu130/CUDA13.0; [snapshot môi trường lịch sử](docs/repository/ENVIRONMENT.json) dùng để truy nguồn, không phải file cài cho mọi máy.

Verifier kiểm size/SHA-256 theo Git index, notebook,14 kiểm tra selector của20,75 score BAF,30 bộ dự đoán nhỏ và bảng tổng hợp. Nó không train, tải dữ liệu, chạy HPO hay bootstrap. `--output` phải nằm ngoài working copy. Chỉ xem báo cáo có thể mở CSV/notebook có output ngay; không cần train.

Cache mặc định nằm trong `.cache/baf-research/` của repo, không yêu cầu ổ D. Có thể đặt `BAF_CACHE_ROOT` tới thư mục ghi được trước khi mở kernel nếu muốn lưu cache ở nơi khác. Source mới có thay đổi đường dẫn cache nên identity advisor khác run lịch sử; manifest/output cũ vẫn nguyên, không bỏ guard để resume bằng source khác.

Mở thư mục `DS317` bằng VS Code, chọn kernel `.venv` vừa tạo. Nếu Jupyter không thấy kernel:

```powershell
& $Python -m ipykernel install --user --name ds317-baf --display-name "DS317 BAF"
```

### Tái lập các run cũ

Cài thành công/import được không bảo đảm tái lập bitwise trên mọi phiên bản. Muốn khớp kết quả cũ phải đối chiếu source/config/data/environment trong manifest của run đó. Bản GitHub không kèm dataset, feature cache hay model để resume.

Notebook05/06 cần snapshot/môi trường cũ như ghi dưới đây. Notebook08 là replay Feedzai riêng: nó tải legacy Optuna/Aequitas vào `external/feedzai08_runtime/packages`; không gộp các gói cũ này vào môi trường mặc định. Các lệnh sau chỉ phục vụ tái lập07/08/09 trong **môi trường riêng** cần đúng profile lịch sử, không phải bước cài bắt buộc cho người clone:

```powershell
# 07: giao diện replay GRU, sau khi đã có môi trường ML/Torch.
& $Python -m pip install "nbformat>=5.10" "ipython>=8.20" "tqdm>=4.66"
# 08: profile tương thích của lần chạy Feedzai đã báo cáo.
& $Python -m pip install "lightgbm==4.7.0" "nbformat==5.11.1" "ipython==9.17.1" "PyYAML==6.0.3" "seaborn>=0.13.2"
# 09: profile của lần chọn class weight đã báo cáo.
& $Python -m pip install "lightgbm==4.7.0" "nbformat==5.11.1" "ipython==9.17.1" "psutil==7.2.2" "matplotlib>=3.8"
```

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

**Campaign mới:** GitHub không có cache/checkpoint để resume run cũ. Dùng working copy riêng, chạy11 tạo cache/campaign từ Base.csv, chép đường dẫn campaign mới in ở11 vào `CAMPAIGN_DIR` của12–16/18/19. Clone đã kèm một campaign kết quả cũ nên sau khi tạo mới sẽ có nhiều campaign; không để bộ chọn tự lấy run lịch sử. Chạy suite tuần tự;16 cần mọi suite đã chọn có FROZEN. Không sửa manifest lịch sử để giả lập resume.

## Nơi xem kết quả

- `results/advisor_combined/`: bảng cuối27 nhóm/81 dòng seed, gồm75 score duy nhất và6 dòng meta tái dùng.
- `results/advisor_step1/`: bảng/audit60 score nghiên cứu gốc.
- `results/research_campaigns/weighting_research_20261028_214637_b9a2294e/`: config, FROZEN, validation summary, selected records; `evaluation/` giữ score/nhãn, bảng paper và bootstrap đã chạy.
- `results/advisor_runs/`: đối chứng CE/sampling và hai run SmallCNN CPU/CUDA riêng biệt,15 bộ dự đoán mỗi run.
- `results/runs/`, `results/comparisons/`: benchmark lịch sử và snapshot source cần thiết.

Chỉ so weighting trong cùng backbone/protocol. Tháng7 đã xem khi phát triển, nên là development test; SD seed không thay CI test. Bootstrap gốc không tự bao phủ18/19. Image-paper là BAF adaptation, LSR có budget giới hạn/chung auxiliary seed42, SmallCNN20 không tái lập đầy đủ paper. Chưa chạy CCRW-E/P hoặc cải tiến đề xuất. Xem [nguồn và giấy phép](THIRD_PARTY_NOTICES.md).
