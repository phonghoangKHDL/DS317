# Bảng khảo sát thực nghiệm trên BAF

Các giá trị là trung bình ± sample SD của các seed được ghi ở bảng chi tiết. AP/ROC-AUC ở thang 0–1; recall, precision, FPR, alert rate ở %; gap FPR ở điểm phần trăm; ratio FPR ở thang 0–1; thời gian ở giây.

TPR@FPR≤5% trên ROC test là độ đo benchmark. Recall/FPR frozen dùng ngưỡng đã khóa trên validation; FPR test có thể khác 5%. Chỉ số nhóm tuổi frozen và các ratio/gap được tính riêng từng seed rồi lấy trung bình. SD seed không phải CI lấy mẫu test.

So sánh weighting trong cùng backbone/feature pipeline. CE batch2048 là baseline loss; đối chiếu meta với CE batch256 sau bổ sung. LSR dùng chung auxiliary seed42. Các run này không phải tái lập nguyên thiết lập paper, không xác lập SOTA toàn cục.

## normalized_head_mlp

| method | ap | recall_at_fpr5_curve | fpr_frozen_threshold | older_fpr | younger_fpr | fpr_gap_older_minus_younger |
| --- | --- | --- | --- | --- | --- | --- |
| ldam | 0.18446 ± 0.00183 | 52.40430 ± 0.57035 | 5.05406 ± 0.36586 | 11.99009 ± 0.32710 | 4.03250 ± 0.37695 | 7.95760 ± 0.18257 |
| ldam_drw | 0.18462 ± 0.00166 | 52.56769 ± 0.74879 | 5.22594 ± 0.20781 | 12.06901 ± 0.46943 | 4.21807 ± 0.20041 | 7.85094 ± 0.42396 |
| norm_ce | 0.19119 ± 0.00394 | 53.15126 ± 0.80760 | 4.94996 ± 0.66954 | 11.61727 ± 0.66840 | 3.96797 ± 0.67855 | 7.64931 ± 0.30466 |

## shared_mlp

| method | ap | recall_at_fpr5_curve | fpr_frozen_threshold | older_fpr | younger_fpr | fpr_gap_older_minus_younger |
| --- | --- | --- | --- | --- | --- | --- |
| balanced | 0.19075 ± 0.00233 | 53.45472 ± 0.57035 | 5.65669 ± 0.29418 | 13.22829 ± 1.12070 | 4.54152 ± 0.37904 | 8.68677 ± 1.33668 |
| cb_ce | 0.19016 ± 0.00260 | 53.75817 ± 0.50660 | 5.40691 ± 0.17797 | 12.39013 ± 0.54131 | 4.37839 ± 0.13431 | 8.01174 ± 0.44001 |
| cb_focal | 0.18861 ± 0.00248 | 53.43137 ± 0.45920 | 5.06349 ± 0.11451 | 11.96288 ± 0.46544 | 4.04733 ± 0.07035 | 7.91555 ± 0.41220 |
| ce | 0.19523 ± 0.00030 | 54.31839 ± 0.42211 | 5.33249 ± 0.14272 | 12.25134 ± 0.55756 | 4.31346 ± 0.11183 | 7.93788 ± 0.52158 |
| drw | 0.19438 ± 0.00111 | 54.36508 ± 0.34544 | 5.35800 ± 0.19938 | 12.42006 ± 0.70840 | 4.31787 ± 0.13825 | 8.10219 ± 0.60775 |
| focal | 0.18213 ± 0.00124 | 51.44725 ± 0.22511 | 4.15623 ± 0.04367 | 10.92062 ± 0.17818 | 3.15995 ± 0.03657 | 7.76067 ± 0.17262 |
| lre | 0.11658 ± 0.00406 | 43.62745 ± 1.21292 | 7.24903 ± 1.56390 | 13.29632 ± 0.26039 | 6.35837 ± 1.78337 | 6.93796 ± 1.73044 |
| mwn | 0.19746 ± 0.00619 | 54.34174 ± 0.55583 | 4.88672 ± 0.30384 | 11.51387 ± 0.63973 | 3.91065 ± 0.30137 | 7.60321 ± 0.59326 |

## lsr_latent_mlp

| method | ap | recall_at_fpr5_curve | fpr_frozen_threshold | older_fpr | younger_fpr | fpr_gap_older_minus_younger |
| --- | --- | --- | --- | --- | --- | --- |
| lsr_nn_balanced | 0.20669 ± 0.00080 | 55.36881 ± 0.26512 | 5.77757 ± 0.11364 | 12.87996 ± 0.26408 | 4.73150 ± 0.09619 | 8.14846 ± 0.19150 |
| lsr_nn_none | 0.21031 ± 0.00202 | 55.15873 ± 1.13854 | 5.71678 ± 0.48888 | 12.76567 ± 0.65842 | 4.67859 ± 0.48253 | 8.08707 ± 0.41847 |
| lsr_nn_weighted | 0.20983 ± 0.00333 | 55.74230 ± 0.64182 | 5.67661 ± 0.58915 | 12.58334 ± 1.12906 | 4.65936 ± 0.51219 | 7.92398 ± 0.63564 |

## lsr_raw_lightgbm

| method | ap | recall_at_fpr5_curve | fpr_frozen_threshold | older_fpr | younger_fpr | fpr_gap_older_minus_younger |
| --- | --- | --- | --- | --- | --- | --- |
| lsr_lgbm_balanced | 0.21519 ± 0.00147 | 56.79272 ± 0.38990 | 4.45632 ± 0.15429 | 10.71652 ± 0.19903 | 3.53430 ± 0.14770 | 7.18222 ± 0.05133 |
| lsr_lgbm_none | 0.21560 ± 0.00169 | 57.93651 ± 0.26512 | 4.34383 ± 0.06353 | 10.72196 ± 0.14725 | 3.40444 ± 0.06487 | 7.31753 ± 0.14692 |
| lsr_lgbm_weighted | 0.21617 ± 0.00218 | 57.30626 ± 0.14577 | 4.37038 ± 0.05525 | 10.73557 ± 0.04320 | 3.43289 ± 0.06587 | 7.30268 ± 0.09305 |

## xgboost_raw

| method | ap | recall_at_fpr5_curve | fpr_frozen_threshold | older_fpr | younger_fpr | fpr_gap_older_minus_younger |
| --- | --- | --- | --- | --- | --- | --- |
| xgb_focal | 0.21783 ± 0.00077 | 57.53968 ± 0.63155 | 4.13003 ± 0.06529 | 10.29744 ± 0.09462 | 3.22167 ± 0.06318 | 7.07577 ± 0.05727 |
| xgb_none | 0.21856 ± 0.00053 | 57.91317 ± 0.25249 | 4.31064 ± 0.09954 | 10.76823 ± 0.23619 | 3.35955 ± 0.08117 | 7.40868 ± 0.16362 |
| xgb_weighted | 0.21488 ± 0.00143 | 57.46965 ± 0.65067 | 4.09125 ± 0.11483 | 10.07973 ± 0.41647 | 3.20924 ± 0.07045 | 6.87049 ± 0.34613 |

