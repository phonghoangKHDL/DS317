# Rà soát repository DS317 / BAF

Ngày rà soát: **06/11/2026, Asia/Saigon**. Phạm vi: `D:\UIT\DS317\BAF_project`.

Chỉ dùng **BAF_project** làm project root để xuất bản. `../BAF_project_audit_20261031_212825_804674/` chứa inventory/log/error/cleanup summary và ZIP báo cáo của lần dọn31/10; không phải mã chạy, dữ liệu hay kết quả bổ sung cần nhập vào repo. Giữ thư mục đó tại máy. Lần rà soát này có hồ sơ/lưu trữ riêng ở `../BAF_repository_review_20261106/`.

## Những gì đã kiểm tra

- Kiểm kê ban đầu **4.266 file dự án, 4.575.634.865 byte (~4,26 GiB)**; thêm 1.320 file metadata Git (root và clone lồng) = **5.586 file**, tổng 4.585.963.026 byte. Không dùng tên/ngày sửa để kết luận dư. Inventory cuối còn bổ sung file báo cáo/script mới; hàng `.git` là snapshot trước rà soát, không phải manifest Git đang biến động. Dung lượng của chính FILE_INVENTORY được ghi sau khi tạo file; hash để trống nhằm tránh tự tham chiếu.
- Đọc tự động2.313 file văn bản/config/source/table, đọc73 notebook gồm source, execution_count, output, error và metadata; kiểm tra AST/import. Đối chiếu thủ công các luồng dữ liệu/protocol/weighting/meta/score/export/resume và notebook chính với manifest/output. Vendor runtime được kiểm kê/hash/scan nội dung, phân tích metadata/import; không tuyên bố review tính đúng từng dòng thư viện bên thứ ba.
- Đối chiếu20 notebook chính: **144 code cell đều có execution_count, 165 output**, không có error output ở bản chính. Đối chiếu notebook con, COMPLETE/FROZEN/SEALED, epoch logs và score; execution_count riêng lẻ không được dùng làm bằng chứng hoàn tất.
- Hash toàn bộ file lúc kiểm kê; tìm trùng nội dung bằng SHA-256 và source notebook. Không có cặp notebook chính trùng source; snapshot trùng có vai trò provenance không bị tự động xóa. Không di chuyển dữ liệu/run/proof vì manifest/cache/resume đang dùng đường dẫn gốc.
- Kiểm tra commit/URL/hash6 nguồn nghiên cứu,12 reference file advisor, giấy phép và NOTICE. Đọc survey Markdown/CSV/JSON; workbook `.xlsx` là artifact khảo sát giữ nguyên byte, không tự viết lại bảng nghiên cứu. PDF/binary/ZIP được kiểm kê/hash và đối chiếu provenance/container; không coi đó là đã đọc/đánh giá toàn văn mọi paper.
- Kiểm tra Git: ban đầu936 file thường và8 gitlink được theo dõi; runtime vendored nằm trong index, gitlink không có `.gitmodules`. Chọn bỏ tracking các checkout/runtime để clone mới tự lấy nguồn theo lock bằng07/08/11. Không xóa checkout trên đĩa và không sửa lịch sử commit.
- Kiểm tra môi trường `ml-gpu`, dependencies tách profile, import thật, đường dẫn và score đã lưu. Không cài/upgrade package, không train lại, không chạy lại HPO/bootstrap nặng.

## Phiên bản nguồn và notebook được chọn

Giữ nguyên byte **src/baf.py, src/gru.py, supplement/reporting.py, supplement/training.py**, cả 5 requirements profile và output của 20 notebook chính. Riêng source chọn run của notebook20 được sửa ở bước hoàn thiện cuối; không chạy lại training và không đổi output lịch sử. Chúng là phiên bản đang được dùng: hash core source khớp campaign11–16, hash4 module khớp advisor manifests. Bản20 hiện tại chứa patch retry CSV trong child worker; không gộp patch đó vào training.py vì sẽ đổi identity/resume của các run.

| Module | SHA-256 |
|---|---|
| src/baf.py | `c6d70e56c4c8fe0aaac2bb2cb5cb75cb1fdd9be6b21baae2326a077026e7cebe` |
| src/gru.py | `ecb673080dfd591a6a1c9116c3f5e78efc0ec3f99932e1d5cf9bd17434d107f6` |
| supplement/reporting.py | `74342fe0e3ece2977629530387ed6da9b6f52ee66d3952b112f4ff809dafae0f` |
| supplement/training.py | `41a93331ec5fceea1676eb0abfd396e596254c1b0427e9eb98d691e6b80462f7` |

| Notebook chính | Cell đã chạy / code cell | Vai trò và lý do giữ |
|---|---|---|
| [01_BAF_class_weighting.ipynb](notebooks/01_BAF_class_weighting.ipynb) | 6/6 | Baseline LR/LightGBM none/balanced; 8 lượt. Giữ output, không thay bằng09 vì khác search budget và protocol. |
| [02_BAF_cost_weighting.ipynb](notebooks/02_BAF_cost_weighting.ipynb) | 6/6 | Cost weighting3 lambda,9 lượt. Khác01 ở trọng số; không phải bản vá. |
| [03_BAF_comparison.ipynb](notebooks/03_BAF_comparison.ipynb) | 5/5 | Audit/so sánh17 prediction của01/02. Notebook đọc kết quả, không train; kết luận validation riêng. |
| [04_BAF_GRU_reproduction.ipynb](notebooks/04_BAF_GRU_reproduction.ipynb) | 5/5 | GRU pilot8192 train,4096 mỗi tập đánh giá,1 epoch. Giữ minh chứng kỹ thuật, không tính benchmark full. |
| [05_BAF_GRU_full.ipynb](notebooks/05_BAF_GRU_full.ipynb) | 6/6 | GRU full seed 1234,2 loss,8 epoch/loss. Thực nghiệm lịch sử; SHA source hiện tại khác; snapshot gốc đã giữ. |
| [06_BAF_GRU_three_seeds.ipynb](notebooks/06_BAF_GRU_three_seeds.ipynb) | 5/5 | Thêm seed 317/2026, tái dùng seed 1234 của05. Không trùng05; cần base run full và source bước06 để tái chạy. |
| [07_BAF_GRU_author_original.ipynb](notebooks/07_BAF_GRU_author_original.ipynb) | 29/29 | Replay GRU nguồn HPO10 + fixed recipe. Khác05/06 ở HPO/RNG; giữ run thành công, run lỗi giữ local. |
| [08_BAF_Feedzai_original.ipynb](notebooks/08_BAF_Feedzai_original.ipynb) | 27/27 | Feedzai Base100 trial, test 6–7/top-five mô tả. Protocol khác09; không dùng top-five test để tuyên bố cải thiện độc lập. |
| [09_BAF_class_weight_selection.ipynb](notebooks/09_BAF_class_weight_selection.ipynb) | 9/9 | Validation-based weighting selection, none/balanced/tuned. Luồng class weighting hiện tại, không thay bảng01/02/08 có split/budget khác. |
| [11_BAF_research_prepare.ipynb](notebooks/11_BAF_research_prepare.ipynb) | 4/4 | Tạo full campaign/cache, source locks. Điểm bắt đầu tái chạy nghiên cứu mới; đã chạy thật, không chỉ kế hoạch. |
| [12_BAF_research_losses.ipynb](notebooks/12_BAF_research_losses.ipynb) | 5/5 | Loss/control: CE/balanced/Focal/CB/DRW/normalized-head/LDAM. 45 COMPLETE predictor fit; giữ training child notebook/log và lựa chọn. |
| [13_BAF_research_meta_weighting.ipynb](notebooks/13_BAF_research_meta_weighting.ipynb) | 5/5 | LRE/MWN với meta holdout trong train. 10 COMPLETE predictor fit; khác họ loss12, không trùng SmallCNN20. |
| [14_BAF_research_LSR.ipynb](notebooks/14_BAF_research_LSR.ipynb) | 5/5 | LSR với latent-NN và LightGBM/control. 30 COMPLETE predictor fit + auxiliary; giữ config/log, model/latent local. |
| [15_BAF_research_Imbalance_XGBoost.ipynb](notebooks/15_BAF_research_Imbalance_XGBoost.ipynb) | 5/5 | Imbalance-XGBoost none/weighted/focal. 15 COMPLETE predictor fit; attempt lỗi thiếu XGBoost và transition repair là provenance. |
| [16_BAF_research_table.ipynb](notebooks/16_BAF_research_table.ipynb) | 6/6 | Đánh giá60 score/20 phương pháp, paper table/bootstrap. Đã SEALED/EVALUATION_COMPLETE; không phải kế hoạch chạy cải tiến. |
| [17_BAF_advisor_metrics.ipynb](notebooks/17_BAF_advisor_metrics.ipynb) | 4/4 | Audit metric đầy đủ và fairness từ60 score. Mở rộng16, có fallback CSV/provenance trên clone; không train. |
| [18_BAF_CE_batch256.ipynb](notebooks/18_BAF_CE_batch256.ipynb) | 3/3 | CE batch256 để khớp nhóm meta. Đã hoàn tất5 fit/3 selected scores; thay thông tin “chưa chạy” trong tài liệu cũ. |
| [19_BAF_sampling_baselines.ipynb](notebooks/19_BAF_sampling_baselines.ipynb) | 3/3 | LightGBM none/balanced/RUS/ROS cùng cache. Đã hoàn tất20 fit/12 selected scores; baseline sampling phục vụ đối chiếu weighting. |
| [20_Original_dataset_weighting_checks.ipynb](notebooks/20_Original_dataset_weighting_checks.ipynb) | 3/3 | CB/LRE/MWN trên MNIST/CIFAR-10 quy mô nhỏ. Bản hiện hành có retry CSV trong worker; giữ patch evidence và failed/successful attempts; không có22. |
| [21_BAF_merge_supplement_tables.ipynb](notebooks/21_BAF_merge_supplement_tables.ipynb) | 3/3 | Ghép bảng gốc +18/19 và nhóm meta đối chiếu. 81 dòng/27 nhóm;75 score riêng +6 dòng tái dùng, không phải81 fit mới. |

`references/gru_author/author_notebook_original.ipynb` là bản tham chiếu tác giả có hash khóa, không phải notebook thay thế07. Sáu `docs/validation/SYNTHETIC_*` là fixture kỹ thuật của11–16, không phải bản trùng thí nghiệm BAF thật. Các notebook đã execute trong results là bằng chứng của worker/proof, khác notebook giao diện; giữ những phần được chọn và giữ toàn bộ bản local.

Đã đọc 50 hàng trong các sheet của workbook `references/survey_20261026/Khao_sat_BAF.xlsx`. Sheet kết quả có 17 lượt **dự kiến**, metric trống và trạng thái “Chưa chạy”: đây là kế hoạch ngày 26/10, không phủ nhận kết quả thực tế đã chạy sau đó. Giữ workbook nguyên bản để truy nguồn, dùng results và notebook có output để lấy kết quả. Các số paper P01–P07 trong survey là thông tin nguồn/khảo sát lịch sử, không tự chuyển thành số nhóm đã tái lập.

## Danh sách chính thức và lý do

Danh sách **từng file** nằm trong [GITHUB_FILES.txt](docs/repository/GITHUB_FILES.txt). [FILE_INVENTORY.csv](FILE_INVENTORY.csv) có path, byte, hash ban đầu/hiện tại, category, include_in_github, reason, action, current_path, duplicate_of, tracked_before, review_method. Duplicate là trùng byte, không đồng nghĩa file thừa; một số bản chính có chủ đích trùng snapshot lịch sử.

| Nhóm chính thức | Quyết định |
|---|---|
| README, README_ADVISOR, README_RESEARCH, notices, .gitignore, .gitattributes | Cài đặt/cấu trúc/thứ tự/kết quả đúng hiện trạng; giữ byte source/evidence qua Git clone |
| src/ và supplement/ | Nguồn hiện hành khớp manifest; không refactor hay đổi dependencies |
|20 notebook chính + child notebook được chọn | Giữ output có giá trị và lịch sử thực thi; failed attempts được đánh dấu riêng |
|5 requirements, install_research.ps1, Install-BAFAdvisorDeps.ps1 | Giữ profile riêng; các helper có tham số interpreter, không thực thi cài đặt trong lần rà soát |
| docs/, references/, workbook khảo sát | Protocol, nguồn phương pháp, hash/commit, validation lịch sử/synthetic, báo cáo/khảo sát, giấy phép |
| results/advisor_step1 và advisor_combined | Bảng gốc/bảng cuối, audit/provenance/group support;75 score duy nhất được phân biệt81 dòng hiển thị |
| research evaluation và advisor CE/sampling evaluation |75 bộ score/nhãn, threshold và metric để kiểm chứng không train |
| original_checks CPU/CUDA |30 bộ probability/nhãn nhỏ, fit records/epoch/summary và8 kiểm tra số học mỗi môi trường; không trộn hai run |
| campaign/suite/fit records, logs, source/data environment hashes |100 predictor fit nghiên cứu,25 fit bổ sung, lựa chọn từ validation, FROZEN/SEALED, repair audit |
| results/runs và comparisons phần nhỏ | Bảng01–09, HPO history/diff/provenance; snapshot source GRU cũ giữ khóa05/06 |
| data/raw/README.md | Hướng dẫn tải đúng Base.csv và hash, không bao gồm dữ liệu |
| scripts/verify_repository.py và docs/repository/ | Công cụ audit không train, môi trường quan sát, kiểm tra thực tế, phạm vi payload và bảng di chuyển |

Phần chọn hiện khoảng **45 MiB** cả inventory, so với4,26 GiB local; không giữ ZIP proof trùng với evidence đã giải nén. Con số chính xác cuối cùng xem inventory/danh sách stage. [EVIDENCE_SCOPE.json](docs/repository/EVIDENCE_SCOPE.json) định nghĩa hash file giữ và danh sách file kết quả chỉ còn local. Index `sha256.json`/COMPLETE/FROZEN lịch sử giữ nguyên có thể nhắc tới payload không được publish: đó không phải lỗi mất file của full run local.

## Những gì chỉ giữ ở máy

- `data/raw/Base.csv`213.427.735 byte, đúng hash đã dùng;5 archive MNIST/CIFAR khoảng182 MB. Dataset có hướng dẫn lấy lại, không đưa vào Git.
- Toàn bộ `external/`:6 repo nghiên cứu, checkout GRU/Feedzai và runtime Aequitas/Optuna/SQLAlchemy. Giữ vị trí để resume; clone mới dùng logic tải/kiểm hash trong notebook 07/08/11. LICENSE/NOTICE cần thiết được giữ riêng trong docs/references.
- Feature cache `results/research_data/` (mảng dense/LSR/row cache/preprocessor), model/checkpoint, latent/diffusion/weights, validation-score arrays, sampled training indices. Metadata/hash index được chọn publish; payload giữ nguyên local.
- `results/proofs/`, mọi ZIP `_review.zip`/`_full_proof.zip`, model/prediction/split payload của benchmark cũ và snapshot lặp trong các suite. Bản GitHub không có đủ payload để resume/verify toàn bộ model.
- Ba run sớm không hoàn tất: `author_gru_original_20261027_203313_ad7415e6`, `feedzai_base_original_20261028_000625_e5053415`, `gru_author_fixed_full_20261027_110618_2707c932`. Đã đối chiếu status/log/partial output, giữ để chẩn đoán, không dùng như benchmark mới. Không di chuyển vì run/log có đường dẫn lịch sử.
- Metadata `.git`, cache interpreter, folder audit31/10 và hồ sơ rà soát06/11. Thư mục cũ không được nhập vào project chính.

## Những gì lưu trữ riêng

Đã chuyển **12 file,139.087 byte** sang `../BAF_repository_review_20261106/archived/`, giữ cây tương đối. [ARCHIVE_MOVES.csv](docs/repository/ARCHIVE_MOVES.csv) ghi từng old/new path, SHA và lý do; bảng đầy đủ đường dẫn tuyệt đối +11 bản copy tài liệu trước sửa ở hồ sơ ngoài project.

| File/nhóm | Cơ sở nội dung và quan hệ sử dụng |
|---|---|
|2 file trong backups/before_research_20261028_214434 | Source trước mở rộng nghiên cứu; nguồn đang import là src/, hash khớp campaign/advisor. Bản backup không phải source resume đang tham chiếu. Giữ snapshot source05/06 cần thiết ở run evidence. |
| BUNDLE_SHA256.json, PACKAGE_SHA256.json, STEP05/06/07/08/09_PACKAGE_SHA256.json | Hash của gói bàn giao trước khi notebook có output/source được mở rộng; đã so với byte trước rà soát, có sai khác. Không dùng làm manifest phiên bản GitHub hiện tại. Xem PACKAGE_HISTORY_CHECKS.json. |
| fix_xgboost_dependency.py và README_FIX_XGBOOST.md | Bản vá một lần trước test; code chủ động từ chối SEALED. Campaign hiện đã SEALED, XGBoost 3.4.1 đã có; giữ repair transition/journal trong results để truy nguồn. |
| scripts/Extract-BAFAdvisor.ps1 | Script cài ZIP cũ dùng Expand-Archive -Force, không nằm trong import graph; không cần sau clone và có thể ghi đè notebook/output. |

Không xóa vĩnh viễn dữ liệu, kết quả, notebook hoặc mã nguồn. Chỉ gỡ tracking bằng Git đối với các file local; các bản trên đĩa vẫn nguyên. Tài liệu được chỉnh có bản sao trước sửa trong `original_documents/` ngoài project.

## Tái lập, môi trường và kiểm tra

Đã chuẩn hóa README/.gitignore theo cây thật; `.gitignore` mặc định bỏ run mới và chỉ mở ngoại lệ cho các file kết quả đã review. `.gitattributes` tắt chuyển đổi newline để SHA source/score/table không đổi sau clone. Nguồn current không đổi, nên không phá source lock đang dùng.

Môi trường quan sát: Python 3.14.3, Torch 2.14.1+cu130/CUDA13.0, RTX 3050 6GB, NumPy 2.5.4, pandas 3.1.0, sklearn 1.9.1, XGBoost 3.4.1. `pip check` đạt; mọi requirement trong 5 profile đều được môi trường hiện tại thỏa. [ENVIRONMENT.json](docs/repository/ENVIRONMENT.json) ghi81 package cùng kết quả kiểm tra. Đây là snapshot môi trường, không tự gộp/đóng pin/cài lại dependencies; không tuyên bố đã dựng môi trường mới trên máy khác.

Kết quả machine-readable: [VERIFICATION.json](docs/repository/VERIFICATION.json), [LOCAL_CHECKS.json](docs/repository/LOCAL_CHECKS.json).

Lần rà soát đầu đã stage **1.105 file, khoảng 45,2 MiB** (số này được thay bằng 1.107 file sau bước hoàn thiện cuối), gỡ 859 entry local/archive khỏi index và loại hết 8 gitlink; dữ liệu trên đĩa vẫn giữ nguyên. Kiểm tra đầu trên bản xuất index đã đạt ở phạm vi evidence. Bước hoàn thiện cuối bổ sung kiểm byte/hash toàn bộ file theo inventory, với 47 notebook, 16 file Python và 1.011 hash evidence; xem báo cáo cuối bên dưới. Đây là kiểm tra bản xuất nội bộ, chưa push GitHub. Các link tương đối trong 6 tài liệu hiện tại đều có đích. Đã đọc danh mục 278 container ZIP/NPZ/XLSX không lỗi; không thấy member tên .env/kaggle.json/credentials.json. Scan lại 994 file text được chọn trước stage không có ứng viên secret.

Git giữ nguyên whitespace của evidence/giấy phép bên thứ ba thay vì sửa nội dung đã có hash. `.gitattributes` nhận CRLF và miễn kiểm tra whitespace cho hai nhóm artifact đó; source hiện hành và tài liệu tự viết vẫn được kiểm tra. Một số log/diff/LICENSE lịch sử có khoảng trắng cuối dòng được giữ có chủ đích.

| Kiểm tra thực hiện | Kết quả |
|---|---|
| Notebook/JSON/Python và file chính thức | Notebook chính, child evidence/synthetic được validate; AST code qua IPython transformer, không execute training |
| Import thật baf/gru/reporting/training trong ml-gpu | Đạt, import đúng file trong project |
|6 repo source lock +12 reference advisor | Commit/remote và32 file research +12 file advisor khớp |
|8 kiểm tra số học CB/LRE/MWN | Đạt; lớn nhất1,2013×10^-13; chỉ bước tính toán nhỏ, không huấn luyện BAF |
| Baseline/cost17 prediction và GRU 3 seed 12 prediction | Audit đọc lại đạt; không chạy lại model |
|75 selected BAF scores | AP/AUC, recall/FPR/precision/alert và age-group metrics khớp ở atol 1e-10 |
| Bảng advisor_step1/combined | Ghép bảng và mean/sample SD khớp;81 dòng =75 score +6 meta reused |
|30 fit SmallCNN CPU/CUDA | Accuracy/balanced accuracy khớp; log-loss CPU khi tính bằng sklearn mới chênh tối đa7,5836e-7, tolerance riêng1e-6; CUDA khớp số lưu |
| Hash phần evidence giữ trên GitHub | Kiểm tra theo EVIDENCE_SCOPE, không giả định đã kiểm checkpoint bị lược bỏ |
| Thông tin đăng nhập | Scan signature token GitHub/OpenAI/AWS, private-key header, credential URL và secret assignment trong 2.313 file text ban đầu: không có ứng viên. File nhạy cảm bị ignore; không publish ZIP/raw data/runtime. Đây là scan pattern, không bảo đảm phát hiện mọi dạng bí mật. |

Không chạy lại huấn luyện/HPO/bootstrap và không cài package. Theo yêu cầu tiếp theo của người dùng, sau khi bản xuất cuối kiểm tra đạt sẽ commit và push nhánh hiện tại lên origin `https://github.com/phonghoangKHDL/DS317`; không force push hoặc viết lại lịch sử. Chỉ stage bằng danh sách chính thức, không dùng git add --all. Các path tracked cũ cần giữ local được gỡ khỏi index bằng danh sách riêng; không rewrite commit cũ.

## Lỗi/giới hạn còn tồn tại

1. **05/06 là thực nghiệm lịch sử, không Run All trực tiếp với source hiện tại.**05 khóa SHA cũ;06 so training signature cả gru.py nên phần research mở rộng làm khác. Snapshot đúng được publish; để train lại dùng working copy riêng/đúng môi trường và full base run local. Không bỏ guard hay sửa nguồn đang khớp campaign chỉ để notebook cũ chạy được.
2. **Manifest lịch sử chứa đường dẫn tuyệt đối và partial payload.** Clone mới đọc/verify bảng được, chưa thể resume campaign cũ vì không có feature cache/checkpoint. Muốn train mới dùng working copy riêng, chạy11 tạo cache/campaign mới, chọn CAMPAIGN_DIR rõ ràng.07/08 yêu cầu Base.csv trước khi phát hiện root;11–21 cần mở từ root/notebooks. Cache/helper trên Windows còn mặc định ổ D như README mô tả.
3. **MWN reference trainer có cú pháp `cuda(async=True)`.** Không parse trên Python hiện tại; giữ nguyên byte theo source lock và license. Code hoạt động dùng port/MetaModule trong resnet.py, không import trainer này. Một JS vendor local không decode UTF-8 (byte0x93); không được publish, không ảnh hưởng import đồ án.
4. **Hai lỗi lịch sử đã có attempt sau thành công.** XGBoost thiếu package và notebook 20 PermissionError ghi epochs.csv vẫn có notebook/log lỗi được giữ. Chúng là provenance, không có nghĩa run cuối thất bại. Không chạy lại repair XGBoost sau SEALED.
5. **Diễn giải khoa học:** month7 đã quan sát, image-paper là adaptation; pilot04/synthetic không phải benchmark. LSR budgeted/chung auxiliary seed 42; không coi3 seed classifier là3 lần fit auxiliary độc lập. Chưa chạy CCRW-E/P hay các paper temporal thiếu source; không đổi kế hoạch thành kết quả.
6. **Giấy phép:** đã bổ sung Apache license cho GRU và giữ giấy phép Feedzai/LDAM/Imbalance-XGBoost cùng CB/LRE/MWN/NOTICE. Checkout LSR không có LICENSE được tìm thấy nên không publish source snapshot LSR. Nhóm chưa chọn giấy phép cho phần tự viết; không tự gán giấy phép cho toàn repo.
7. **Tái lập cross-environment:** giữ raw float32 score/nhãn và số đo gốc; log-loss có sai số nhỏ giữa sklearn 1.8 và1.9 như trên. Đã kiểm môi trường hiện tại, chưa dựng lại trên máy mới hoặc kiểm lại toàn văn từng paper/thư viện vendor.


## Hoàn thiện cuối trước commit/push

- **Notebook20:** xóa ưu tiên `original_checks_582e4a8ada1a` trong code chọn run. `RESUME_RUN = None` chỉ dùng identity hiện tại (config/source/environment); explicit resume cần manifest hiện hữu khớp toàn bộ identity, kể cả field dư. Giữ guard đường dẫn canonical để khớp `reporting.launch` và từ chối trước mọi launch khi run đã bị chuyển chỗ. Chạy 14 fixture nhỏ từ chính AST của hàm chọn run; không execute setup/worker/train. Output và execution_count cũ giữ nguyên; chúng thuộc source trước sửa selector, không được trình bày như lần train mới của code đã sửa.
- **CSV khảo sát:** xác nhận working tree gốc 6.673 byte, 8 CRLF, hash `aa0e443c862f6761c784a92c517fe848ea2afe4206d0f6670e11a94d2ce9aa79`. Index/bản xuất cũ là 6.665 byte do LF normalization; đã stage lại byte gốc dưới `.gitattributes` `-text`. Không sửa `references/survey_20261026/metadata/artifact_hashes.json` hoặc hash lịch sử khác để che sai lệch. Đây là lỗi phạm vi kiểm tra cũ: evidence hash pass không bao phủ CSV tham chiếu.
- **Kiểm inventory:** kiểm size và SHA-256 của toàn bộ 1.107 file chính thức; 1.106 file có hash bắt buộc. Riêng FILE_INVENTORY.csv có dung lượng đã ổn định, SHA riêng trống được validator kiểm tra rõ ràng; không miễn hash cho file nào khác. Danh sách official trong inventory phải khớp GITHUB_FILES.txt và không có duplicate row. Báo cáo `--output` phải ở ngoài working copy để tránh tự làm hỏng hash đã kiểm.
- **Kiểm hồi quy nhẹ:** 8 fixture phát hiện CRLF→LF, đổi nội dung cùng kích thước, thiếu hash, hash giả tự tham chiếu, official-list khác nhau, duplicate row và sai dung lượng chính inventory. Các kiểm score/bảng/notebook/import có sẵn được giữ nguyên.
- **Bản cuối:** thêm 2 script kiểm tra nhẹ; danh sách chính thức có 1.107 file. Làm mới inventory và VERIFICATION.json, stage chỉ các đường dẫn chính thức bằng danh sách, stage lại byte dưới quy tắc newline hiện tại, xuất lại từ index và chạy verifier trên chính bản xuất. Báo cáo cuối kèm inventory bảo vệ cả file script, README, notebook20, CSV và báo cáo kiểm tra. Origin/nhánh được kiểm tra trước commit/push; kết quả commit/push và hash commit được ghi ở phản hồi cuối cùng, tránh tạo báo cáo tự chứa hash commit của chính nó.

Manifest, score, log, notebook executed, CSV-retry patch và mọi evidence trong các run cũ giữ nguyên byte. Source bốn module và dependencies không thay đổi. Không thực thi HPO, bootstrap hoặc huấn luyện.
