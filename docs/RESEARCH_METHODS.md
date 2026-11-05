# Khảo sát thực nghiệm các phương pháp trọng số trên BAF

## Phạm vi và cách gọi kết quả

Gói này tạo **kết quả do người dùng chạy lại cơ chế phương pháp trên BAF**, không xác nhận đã tái lập con số gốc trên CIFAR/MNIST/các tập tabular khác. Cột “mức độ tái lập” phải ghi `BAF transfer / adapted implementation`; LSR ghi thêm `budgeted`. Chưa chạy full BAF thì không điền kết quả vào thuyết minh. Chế độ pilot và toàn bộ proof SYNTHETIC chỉ dùng kiểm tra kỹ thuật.

Đây là bước khảo sát phương pháp đã công bố để lập bảng research và tìm khoảng trống. Gói **chưa chạy phương pháp cải tiến đề xuất**, chưa khẳng định CCRW-E hoặc một tổ hợp nào có tính mới. LDAM là loss điều chỉnh margin, được giữ làm đối chứng lân cận; không gọi riêng LDAM là sample weighting. Các method/ablation từ cùng paper không tính thành nhiều paper độc lập.

## Nguồn và mức chuyển giao

Nguồn tác giả được khóa bằng URL, commit và SHA-256 từng file trong `research_sources.json`. Notebook 11 clone đúng commit; lúc chạy giữ snapshot các file được sử dụng trong evidence. Focal gốc: Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017, https://arxiv.org/abs/1708.02002. Phương trình alpha_t(1-p_t)^gamma CE được chuyển sang binary classification bằng hai logits; không chạy detector RetinaNet.

| Paper / nguồn | Phương pháp trong gói | Giữ cơ chế | Điều chỉnh phải ghi trong báo cáo |
|---|---|---|---|
| Focal Loss, ICCV 2017 | focal | alpha_t, gamma=2, focal CE | BAF MLP, 3 alpha; không phải detector gốc |
| Class-Balanced Loss Based on Effective Number of Samples, CVPR 2019; richardaecn/class-balanced-loss | cb_ce, cb_focal | effective-number weight, chuẩn hóa tổng 2 lớp bằng 2; CB-Focal dùng 2 sigmoid như file tác giả | Port TensorFlow sang PyTorch; MLP thay CNN; giữ scale loss theo tác giả |
| Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss, NeurIPS 2019; kaidic/LDAM-DRW | drw, ldam, ldam_drw | margin n_y^-1/4, max .5, scale 30; DRW beta .9999 ở 20% epoch cuối | CUDA hardcode được thay bằng device hiện tại; lịch 160/200 chuyển thành tỷ lệ 80%; giữ head chuẩn hóa và baseline norm_ce có scale 30 |
| Learning to Reweight Examples for Robust Deep Learning, ICML 2018; uber-research/learning-to-reweight-examples | lre | epsilon=0, một bước SGD ảo, meta-gradient, clamp và normalize; tất cả weight=0 được giữ đúng | TF1 sang torch.func; meta holdout cân bằng lấy bên trong train BAF; MLP; không dùng future validation làm meta |
| Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting, NeurIPS 2019; xjtushujun/Meta-weight-net_class-imbalance | mwn | WeightNet 1→100→1 sigmoid; meta update qua virtual SGD; batch normalization của weights như code | MLP thay ResNet; predictor LR theo lưới; VNet SGD 1e-5, momentum .9, nesterov; thời lượng và lịch LR giới hạn |
| Latent Score-Based Reweighting for Robust Classification on Imbalanced Tabular Data, ICML 2025; YunzeTong/latent-score-based-reweighting | lsr_nn_weighted, lsr_lgbm_weighted | import trực tiếp VAE/EDM/diffusion class tác giả; error_diff, T=3, 10 sigma ×32 noise; mean weight=1 | train-only BAF encoding; meta nội bộ làm VAE validation; capped 30/50 epoch; streamed reduction; LightGBM là classifier transfer bổ sung |
| Imbalance-XGBoost: Leveraging Weighted and Focal Losses for Binary Label-Imbalanced Classification with XGBoost; jhwjhw0123/Imbalance-XGBoost | xgb_weighted, xgb_focal | import nguyên file tác giả tính gradient/Hessian | dùng xgb.train API hiện tại, bỏ wrapper sklearn cũ; logitraw + sigmoid; budget 300 cây |

Các đối chứng `ce`, `balanced`, `norm_ce`, `lsr_nn_none/balanced`, `lsr_lgbm_none/balanced`, `xgb_none` không được ghi thành các paper mới. Chỉ so hiệu ứng trong cùng `comparison_group`. LRE/MWN so với CE cùng MLP, train rows, seed, khởi tạo và validation rule. Nhóm normalized-head so với `norm_ce`. LSR latent MLP so với latent MLP cùng encoder; LightGBM và XGBoost có nhóm riêng. Không xếp chung để kết luận loss tốt nhất chỉ từ AP của hai backbone khác nhau.

## Protocol khóa trước khi chạy

- BAF Base.csv SHA-256: `7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809`, 1.000.000 hàng, month 0..7.
- Train tháng 0..5. Lấy 128 fraud +128 legit ở tháng 5 bằng seed317 làm meta holdout, **loại khỏi train của tất cả learner**. Validation tháng6, test tháng7. Vì train đã thay đổi, không chép AP notebook09 vào cùng bảng như một run của protocol này.
- Không dùng month/label làm feature. Giữ customer_age làm feature và báo nhóm `<50` / `>=50`. Sentinel được xử lý bằng quy tắc BAF cũ. Dense MLP/tree: median, standardization numeric và one-hot categorical, unknown→0; LSR: train-only quantile numeric và ordinal categorical, unknown index0. Mọi bộ biến đổi chỉ fit core train, không fit meta/validation/test.
- HPO dùng seed42, 3 recipe mỗi method; chọn AP validation cao nhất, tie chọn candidate index nhỏ hơn. Hai seed317,2026 dùng cấu hình đã chọn. Run seed42 thắng HPO được reuse, không fit lại. Mặc định 20 method ×(3+2)=100 fit dự đoán, **không phải 100 Optuna trial cho mỗi method**. LSR có thêm 1 VAE+2 diffusion model, dùng auxiliary seed42 chung cho các seed classifier.
- CE/balanced/LRE/MWN/normCE/LDAM: LR {0.01,0.03,0.1}. CB beta {0.9999,0.99999,0.999999}, LR .03. Focal alpha {0.25,0.75,0.99}, gamma2, LR .03. LSR-NN LR {.0003,.001,.003}, T3; LSR-LGBM leaves {15,31,63}. XGB: 3 **joint recipes**, depth {3,5,7}, alpha=IR^{.25,.5,1} cho weighted; gamma2 cho focal. Đây là search budget giới hạn, không phải full factorial hay đảm bảo hội tụ.
- Neural tối đa20 epoch, SGD + momentum .9/nesterov/weight_decay1e-4; LSR predictor Adam. Early stop/chọn checkpoint theo **unweighted validation AP**, patience5; DRW vẫn phải được huấn luyện đến giai đoạn weight bật. Không áp dụng weight vào nhãn validation để chọn cấu hình.
- LGBM reg_lambda1, min_child_samples100; XGB lambda1, histogram, eta.05. Weight thay đổi gradient/Hessian và tác động regularization: bảng này không tuyên bố đã loại hết tương tác bằng một recipe cố định. Nếu nghiên cứu tiếp cần kiểm tra độ nhạy regularization sau khi chọn hướng.
- Khóa tất cả suite đã chọn và ngưỡng từ validation tại FPR≤5% trước bước test. Test month7 của Base đã được nhìn ở notebook cũ: **development test**, không gọi confirm/untouched. Xác nhận phương pháp đề xuất sau này cần tập/cửa sổ giữ riêng.
- Báo AP, ROC-AUC, recall ở ngưỡng khóa từ validation, **FPR thực tế trên test** và FPR/TPR từng nhóm tuổi. `recall_at_fpr5_curve` là metric tính trực tiếp trên test ROC, phải phân biệt với recall ở ngưỡng đã khóa; bảng chi tiết lưu cả hai.
- Paired IID-row bootstrap 1.000 lần cho ΔAP so với baseline cùng backbone, cùng hàng cho mọi cặp seed; CI có điều kiện trên các model đã fit và HPO đã chọn. SD theo seed là chỉ số khác. CI không mô hình hóa drift tháng hay nhiều phép so sánh, không khẳng định độc lập về nhân quả.

## Chi tiết LSR cần đọc

LSR là suite nặng. Source VAE có cap4.000 epoch, diffusion10.001; bản BAF mặc định30/50 **chưa chứng minh hội tụ**. Giữ lịch sigma10, Monte Carlo32; tính trước khoảng `2*N_train*10*32` mẫu denoising. Streaming chỉ giảm RAM/disk, không làm biến mất chi phí tính. Có thể chọn bỏ LSR từ lúc tạo campaign để hoàn thành bảng các phương pháp còn lại; không thêm/bớt sau khi mở test cùng campaign.

Code tác giả center diffusion train theo từng lớp nhưng center khi score bằng mean toàn train. Mặc định `lsr_centering='author'` giữ đúng hành vi này; lựa chọn `shared` là một biến thể đã thay cơ chế và phải gắn nhãn riêng. Error được lấy mean trên chiều latent/noise/sigma và EDM-weight như nguồn. Streaming giữ phép mean nhưng thứ tự RNG khác tensor allocation gốc; dùng logsumexp để tránh exp overflow. VAE reconstruction dùng sample latent như nguồn, selection qua meta holdout thay vì future test. Auxiliary seed chỉ có một: SD cuối là biến thiên classifier có điều kiện trên encoder/weights, không phải ba lần fit LSR độc lập.

## Bảng research và khoảng trống

Notebook16 xuất kết quả thật, không tự điền claim “SOTA trên BAF”. Mỗi dòng cần: paper/năm, cơ chế, dữ liệu paper gốc (đọc PDF), implementation/adaptation, backbone, split, search budget, seed, AP/CI, fairness, chi phí và hạn chế. Dùng `research_table.csv`, `test_metrics.csv`, `paired_bootstrap.csv`, provenance và proof; giữ cột kết quả gốc của paper tách với kết quả BAF do mình chạy.

CSALBGRU, CALMID, MUSCATEL, DBGR và các bài temporal/density gần đề xuất còn cần paper/author code để đối chiếu. Gói không giả lập chúng bằng một decay tùy ý và không nhận là đã tái lập. Sau các run này vẫn cần literature về **temporal/class-conditional relevance và ESS** để đánh giá khoảng trống. Việc một phương pháp chưa được chạy trong gói không chứng minh chưa ai nghiên cứu.

Chưa cần chạy CCRW-E/P trong bước khảo sát này. Chỉ sau khi đọc paper, xem bảng kết quả/proof và rút ra câu hỏi cụ thể mới viết đóng góp **dự kiến**, đối chứng cơ chế, ablation, split xác nhận và tiêu chí chấp nhận trong thuyết minh.
