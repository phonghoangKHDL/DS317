# Khảo sát phương pháp đánh trọng số trên BAF và danh sách chạy

Ngày kiểm tra: 26/10/2026. Phạm vi: 7 công trình có thực nghiệm BAF, trong đó đã kiểm tra snapshot mã nguồn liên kết với 5 công trình. Đây là khảo sát có mục tiêu, chưa phải tổng quan hệ thống đầy đủ. Chưa có kết quả BAF do phiên này tự huấn luyện.

**Quyết định:** bắt đầu với LR và LightGBM không trọng số/balanced; bổ sung cost weighting từ DeCCaF. GRU weighted/unweighted chạy sau pilot tài nguyên. Chưa chọn toàn hệ TEMPLAR/VR-FraudNet. Hiện chưa đủ bằng chứng để gọi bộ đối chứng này là toàn bộ SOTA trọng số trên BAF.

## 1. Bảng khảo sát paper

| ID | Công trình | Cơ chế | Thiết lập BAF | Vai trò |
|---|---|---|---|---|
| P01 | [Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation](https://proceedings.neurips.cc/paper_files/paper/2022/hash/d9696563856bd350e4e7ac5e5812f23c-Abstract-Datasets_and_Benchmarks.html) (2022) | LightGBM mốc; không phải thuật toán trọng số mới | Train month 0–5; test 6–7 | Mốc BAF |
| P02 | [Objective over Architecture: Fraud Detection Under Extreme Imbalance in Bank Account Opening](https://doi.org/10.3390/computation13120290) (2025) | LR/SVM/RF/LightGBM: None hoặc balanced; GRU: weighted cross-entropy | Ngẫu nhiên phân tầng 80/20; GRU tách validation từ train | Trọng số trực tiếp |
| P03 | [Cost-Sensitive Learning to Defer to Multiple Experts with Workload Constraints](https://research.feedzai.com/oshuzuxa/2024/08/Alves_Cost_Sensitive_Learning_TMLR_2024.pdf) (2024) | Weighted log-loss: w(y=0)=lambda; w(y=1)=1 | Alert model train 0–2, val 3; classifier h code train 3–5, val 6, test 7 trên alerts | Trọng số theo chi phí |
| P04 | [TEMPLAR fraud a verifier grounded calibrated and cost sensitive framework for transaction fraud detection](https://doi.org/10.1007/s44248-026-00117-y) (2026) | Focal loss ở encoder; thêm routing, stacking, calibration, chọn ngưỡng chi phí | Theo thời gian 70/15/15; future slice 10% cuối | Hệ nhiều thành phần |
| P05 | [Grounding language models with deterministic verifiers for fraud detection](https://doi.org/10.1007/s44163-026-02355-0) (2026) | LightGBM triage dùng focal + class-balanced; có cost weighting | Theo thời gian; pipeline có graph/LLM và calibration | Hệ nhiều thành phần |
| P06 | [The Explainability–Reliability Gap in Fraud Detection: Evidence from SHAP and Permutation Importance Under Distribution Shift](https://doi.org/10.3390/fintech5030077) (2026) | Logistic Regression balanced; trọng tâm là explanation reliability | Within-variant/cross-variant; mô tả chưa đủ để chốt mọi tỷ lệ chia | Đối chiếu BAF |
| P07 | [Reinforcement-guided hyper-heuristic hyperparameter optimization for fair and explainable spiking neural network-based financial fraud detection](https://doi.org/10.1016/j.knosys.2026.115814) (2026) | SNN population coding và tối ưu hyperparameter; trọng tâm không phải class weighting | Bản arXiv v3 ghi 90/10; khác temporal split nhóm | Ngoài hướng trọng số |

## 2. Kết quả kiểm tra nguồn

### P01

Đối chiếu: Paper §3.2, §4, Fig.2; notebooks/empirical_results.ipynb cells 8,10.

Notebook giữ month trong X và tính điểm ROC trên test. Đó là metric xếp hạng, khác ngưỡng chọn ở validation rồi áp dụng cho tương lai.

Kết quả công bố: 100 cấu hình LightGBM; hình 2, không dùng một số tổng hợp làm SOTA.

Quyết định: Chọn làm mốc tái lập riêng; không xếp số paper cạnh month-7 của nhóm.

Mã nguồn: [feedzai/bank-account-fraud](https://github.com/feedzai/bank-account-fraud/tree/3377cfd8fc20edcda3dd9dd14c10ed3dd276c6de), commit `3377cfd8fc20edcda3dd9dd14c10ed3dd276c6de`.

### P02

Đối chiếu: Paper §2.2, §2.4, Tables 1–3; notebook cells 3,29,30 (đếm từ 0).

Đã thấy balanced và CrossEntropyLoss(weight=...) trong notebook. Danh sách feature code giữ source và bỏ prev_address_months_count; mô tả paper khác. Cần cố định feature khi chạy chung.

Kết quả công bố: GRU recall 0.78, precision 0.05; không phải Recall@FPR5%.

Quyết định: Chọn LR/LightGBM và GRU weighting; ghi rõ bản tái lập và bản đổi temporal split.

Mã nguồn: [wn-nwN/fraud-detection-project-Nov2025](https://github.com/wn-nwN/fraud-detection-project-Nov2025/tree/6e8252cb9080378905b2aae30033c5946b72335b), commit `6e8252cb9080378905b2aae30033c5946b72335b`.

### P03

Đối chiếu: Paper §3, §4.1, §4.3, Table 3; Code/classifier_h/training.py:55–69; hpo_wce.py:152.

Code dùng sample_weight và weighted validation log-loss. Lambda={0.0114,0.057,0.285}. Áp dụng lên toàn bộ Base là chuyển giao thành phần; không tái lập đầy đủ hệ DeCCaF.

Kết quả công bố: Giảm chi phí trung bình 8.4% của hệ DeCCaF; không phải tăng recall 8.4 điểm.

Quyết định: Chọn cơ chế cost weighting làm đối chứng; giữ thử nghiệm nguyên alerts riêng nếu tái lập paper.

Mã nguồn: [feedzai/deccaf](https://github.com/feedzai/deccaf/tree/b910b48d7e146d53b5f145224b819bdb0962d40a), commit `b910b48d7e146d53b5f145224b819bdb0962d40a`.

### P04

Đối chiếu: Paper §3; src/templar_fraud/models/saint.py:129; data/datasets.py:39–51.

Code giới hạn encoder tối đa 30 epochs. Có fallback encoder ngẫu nhiên khi thiếu torch. Future slice nằm trong test: không phải tập độc lập. Chưa giải thích được số fraud 10,290 trong paper so với Base nhóm 11,029.

Kết quả công bố: BAF AUC-PR 0.498 theo paper; chưa tự chạy.

Quyết định: Giữ tài liệu liên quan; chưa chọn chạy toàn hệ trong vòng đầu.

Mã nguồn: [raihandinaribn/templar-fraud](https://github.com/raihandinaribn/templar-fraud/tree/5ae9f6c76d20cd12a2321bcdfbb7e17dc42ed9d2), commit `5ae9f6c76d20cd12a2321bcdfbb7e17dc42ed9d2`.

### P05

Đối chiếu: Paper §3.2, §4.6; README phần Exact numerical reproduction; src/vrfraudnet/losses/.

Repo tự ghi không thể tái lập đúng số paper: thiếu adapter/corpus Stage 2 và một số thiết lập. Đã thấy focal, effective-number weights và cost weights; không xem demo là bằng chứng BAF.

Kết quả công bố: AUPRC D1 0.5247 theo paper; chưa tự chạy.

Quyết định: Chỉ cân nhắc triage focal+CB làm baseline thành phần sau kiểm tra; không lấy số toàn hệ làm mục tiêu đã xác nhận.

Mã nguồn: [abdulbasit12312/VR-FraudNet](https://github.com/abdulbasit12312/VR-FraudNet/tree/7f552994bde0efe4e6706ceccdfb398d8e6450d0), commit `7f552994bde0efe4e6706ceccdfb398d8e6450d0`.

### P06

Đối chiếu: Paper §3.6, §3.11, §4.7; Data Availability.

PDF xác nhận balanced LR. Chưa xác minh được URL code tác giả; Data Availability ghi upon request.

Kết quả công bố: Không trích một điểm làm SOTA trọng số.

Quyết định: Giữ nguồn về balanced LR và kiểm tra khác biệt variant; chưa chọn làm paper tái lập chính.

### P07

Đối chiếu: Publisher; arXiv:2508.16915v3 §4.1.2 và Data and Code Availability.

Đã xác minh bản xuất bản KBS; đọc full text arXiv v3. Chưa xác minh URL repo. Không mặc định 90/10 tương đương chia theo month.

Kết quả công bố: Recall 90.8% là số tác giả báo cáo; chưa tự chạy.

Quyết định: Giữ soát kết quả BAF rộng; không chọn cho hướng trọng số vòng đầu.

## 3. Danh sách chạy đã chốt

| ID | Mô hình | Trọng số/loss | Nguồn | Khi chạy |
|---|---|---|---|---|
| B01 | Logistic Regression | Không trọng số | P02/P06 | Vòng 1 |
| B02 | Logistic Regression | Balanced: N/(2*n_c) | P02/P06 | Vòng 1 |
| B03 | LightGBM | Không trọng số | P01/P02 | Vòng 1 |
| B04 | LightGBM | Balanced: N/(2*n_c) | P02 | Vòng 1 |
| B05 | LightGBM | Cost: w0=lambda,w1=1; lambda=0.0114/0.057/0.285 | P03 | Vòng 1 mở rộng |
| B06 | GRU tác giả | Cross-entropy không trọng số | P02, ablation nhóm bổ sung | Sau pilot tài nguyên |
| B07 | GRU tác giả | Balanced weighted cross-entropy | P02 | Sau pilot tài nguyên |
| B08 | LightGBM triage | Focal + effective-number CB | P05, thành phần | Tùy chọn sau vòng 1 |

Đây là các cấu hình đối chứng, không phải tám thuật toán SOTA độc lập. Vòng 1 dùng tham số cố định để kiểm tra pipeline. Kết luận cuối cần tuning tương đương giữa các phương pháp. Trong cùng một mô hình, giữ nguyên feature/preprocessing; chỉ thay cơ chế trọng số. Không trộn số toàn hệ với số của một thành phần.

## 4. Hai thiết lập thực nghiệm

**Tái lập tác giả:** giữ dataset/version, feature, split, loss, HPO, metric và môi trường công bố. Chạy trong thư mục mới, tránh vô tình nạp checkpoint hoặc kết quả có sẵn. Riêng DeCCaF phải phân biệt toàn Base với tập alerts và hệ có chuyên gia. Script gói này không phải bản tái lập nguyên ba paper.

**So sánh chung của nhóm:** BAF Base đầy đủ; train month 0–5, validation 6, đánh giá cuối month 7. Đó là thiết lập hiện tại của nhóm, khác original BAF train 0–5/test 6–7. Month 7 đã được xem trong notebook cũ nên phải công khai đây là đánh giá thăm dò trên holdout từng được kiểm tra. Có thể thêm các cửa sổ forward validation trong tháng 0–6 để kiểm tra tính ổn định; không gọi việc đó là tạo thêm tập test độc lập.

Trọng số và mọi phép fit tiền xử lý chỉ lấy từ train. Theo datasheet: năm cột sentinel -1 được đổi thành missing, và mọi intended_balcon_amount âm là missing; credit_risk_score âm vẫn hợp lệ. Script dùng cùng các quy tắc cho mọi trọng số trong từng backbone. Không dùng đồng thời class_weight, scale_pos_weight và sample_weight.

**Hai cách đo cần đặt tên riêng:**

- `recall_at_fpr5_curve`: điểm ROC trên chính tập được báo cáo, có dùng nhãn tập đó để xác định điểm; đây là metric xếp hạng chuẩn, không phải ngưỡng triển khai được chọn trước.
- `recall/fpr_frozen_threshold`: chọn ngưỡng trên validation thỏa FPR ≤ 5%, rồi áp dụng nguyên sang test. FPR test có thể vượt 5% khi phân phối thay đổi.
- Báo cáo AP (average precision, không tự đồng nhất với mọi định nghĩa AUC-PR), ROC-AUC, precision, alert rate, confusion matrix, fit time và bộ nhớ. FPR 5% không đồng nghĩa review 5% tổng hồ sơ.

Trong vòng tuning, có thể dùng tháng 0–4 để fit và tháng 5 để chọn cấu hình, sau đó refit 0–5 và dành tháng 6 hiệu chỉnh ngưỡng. Phải ghi khác biệt so với các kết quả notebook cũ. Pilot trong gói chưa thực hiện HPO và dùng validation 6 cho cả kiểm tra phương pháp/ngưỡng; số validation vì vậy là số phát triển, không là đánh giá độc lập.

## 5. Chạy vòng đầu bằng notebook hoặc Python

Mở `notebooks/01_BAF_class_weighting.ipynb` để chạy từng bước. Notebook và CLI dùng chung các hàm trong `code/run_baselines.py`. Dữ liệu đặt ở `data/raw/`; kết quả thật lưu ở `results/runs/`; tài liệu khảo sát ở `docs/`; nguồn paper/code và metadata ở `references/`. Trên Windows, chọn **Extract all** trước khi mở project và chọn đúng Python kernel.

Các lệnh bên dưới chạy từ thư mục project root.

Gói chưa chứa Base.csv. Cần tệp đầy đủ từ [nguồn Feedzai](https://github.com/feedzai/bank-account-fraud) / Kaggle chính thức. Bảng Fact_Application phục vụ BI không thay thế được benchmark đầy đủ. Script yêu cầu đủ 1 triệu dòng và schema Base; ghi SHA-256 để xác định đúng bytes.

```bash
python -m pip install -r requirements.txt
python code/check_baf_benchmark.py
python code/run_baselines.py --data "D:/BAF/Base.csv" --output "results/runs/pilot_01" --threads 4
```

Lệnh mặc định chỉ đánh giá validation tháng 6, chưa tính điểm tháng 7. LR chạy một seed vì solver xác định. LightGBM chạy 42, 317, 2026. Mỗi thư mục output phải mới để không ghi đè log.

Thêm đối chứng cost weighting:

```bash
python code/run_baselines.py --data "D:/BAF/Base.csv" --output "results/runs/cost_01" --methods lgbm_cost_0.0114 lgbm_cost_0.057 lgbm_cost_0.285 --threads 4
```

Cost weights giữ scale gốc w0=lambda,w1=1; balanced dùng N/(2*n_c). Hai cách có thể khác cả phân bố trọng số và tổng độ lớn; cần ablation chuẩn hóa nếu muốn quy kết tác dụng. Đưa cost weighting từ alerts sang toàn Base là adaptation được ghi rõ, không phải chạy lại đầy đủ DeCCaF.

Khi đã cố định phương pháp và cấu hình từ development, thêm `--evaluate-test` vào lệnh và dùng output mới. Script ghi manifest, metrics.csv và file dự đoán trong predictions/ theo method/seed/partition. Giữ nguyên các file dự đoán để bootstrap có ghép cặp. Không dùng kết quả test để tiếp tục tuning rồi gọi là final test.

`requirements.txt` là môi trường pilot hiện đại; môi trường tác giả xem từng repo. Mã hiện tại chỉ cài LR/LightGBM và cơ chế balanced/cost. GRU và focal+CB được ghi kế hoạch, chưa có adapter trong gói.

## 6. Điều kiện để chuyển sang đề xuất cải tiến

1. Tái lập được ít nhất các đối chứng phù hợp đã chọn; giải thích được sai lệch với kết quả paper khi khác split hoặc metric.
2. Có kết quả tuning theo cùng ngân sách, báo cáo seed và so sánh ghép cặp trên cùng hồ sơ. Seed variability không thay cho bất định do hữu hạn dữ liệu; LR xác định không có ba lần chạy độc lập chỉ nhờ đổi seed.
3. Chứng minh một hạn chế cụ thể còn tồn tại, chẳng hạn trọng số cân bằng không ổn định giữa các cửa sổ thời gian sau khi đã so cùng FPR/ngưỡng. Đây là câu hỏi cần kiểm tra, chưa phải phát hiện.
4. Rà literature lần nữa theo hạn chế quan sát được rồi thiết kế cơ chế mới, ablation và đo tài nguyên. Áp dụng một loss cũ lên BAF không tự tạo ra tính mới.

Focal Loss gốc, CB Loss, Meta-Weight-Net và LSR được để ở tài liệu nền. Lượt trước chưa xác minh chúng đã benchmark trên BAF nên không đưa số dataset khác vào bảng SOTA BAF.

## 7. Dấu vết kiểm chứng

- `references/metadata/paper_manifest.json`: URL nguồn và SHA-256 của 7 PDF đã tải/đọc.
- `references/metadata/repository_manifest.json`: commit và hash từng tệp mã nguồn đã kiểm tra của 5 repo.
- `results/checks/source_checks.json`: các kiểm tra thành phần/static audit thực sự chạy trong phiên này; không phải kết quả huấn luyện BAF.
- `docs/Khao_sat_BAF.xlsx`: bảng khảo sát, danh sách chạy và bảng trống nhập kết quả tự chạy.
- `results/templates/planned_runs.csv`: 17 lần chạy validation đã lên lịch, mọi metric còn trống. Không điền số paper vào cột nhóm tự chạy.

Notebook và CLI ghi dự đoán vào `results/runs/<run_id>/predictions/`. Nếu không đo được RSS do giới hạn môi trường, giá trị để trống và ghi lý do ở `memory_warning`; không coi ô trống là 0 MiB.
