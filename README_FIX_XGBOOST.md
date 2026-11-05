# Sửa thiếu XGBoost cho campaign11–16

1. Thực hiện khi các notebook khác đã xong/dừng, không đang huấn luyện. Đóng kernel15.
2. Giải nén vào D:\UIT\DS317, thêm script vào BAF_project; không ghi đè src/notebook/results.
3. PowerShell:

```powershell
& "D:\UIT\.venvs\ml-gpu\Scripts\python.exe" `
  "D:\UIT\DS317\BAF_project\fix_xgboost_dependency.py"
```

Nếu có nhiều campaign, thêm --campaign "đường dẫn campaign đã dùng12–15". Script liệt kê các đường dẫn khi chưa chọn được; không chọn newest tùy ý.

Script cài xgboost>=3.1,<4 bằng đúng interpreter, chỉ wheel, --no-deps; cache/temp nằm trong project/results/dependency_cache trên D. Không đổi Torch, numpy hoặc scipy. Nếu không có wheel phù hợp, pip báo lỗi và script chưa chuyển manifest.

Khóa môi trường của bản cũ ghi XGBoost là not installed. Sau khi thêm thư viện, script lưu manifest trước sửa cùng môi trường huấn luyện gốc; cập nhật môi trường resume, signature, liên kết FROZEN và chỉ mục hash. Không viết lại lịch sử các fit như thể đã chạy với XGBoost mới. Audit nằm trong evidence/dependency_repairs, backup+journal nằm trong campaign/dependency_repairs. Các ZIP proof cũ vẫn giữ nguyên. Full proof xuất sau này sẽ có audit. Chưa cần xuất lại proof12–13 ngay.

Chỉ cho phép thêm XGBoost còn thiếu; không bỏ qua kiểm source, không chấp nhận thay đổi dependency khác, không sửa sau khi test đã SEALED. Nếu gặp trường hợp đó, giữ nguyên file và gửi thông báo lỗi để đối chiếu.

Khi báo READY: chọn đúng kernel UIT ml-gpu, Restart Kernel rồi Run All15 với CÙNG campaign. Không tạo lại campaign ở11; không chạy lại12–13; không xóa attempt bị lỗi.

Kiểm thử local: fixture với một suite frozen và một suite XGB bị lỗi. Đã kiểm model/score không đổi, môi trường gốc được giữ, hash+FROZEN khớp, rerun idempotent, từ chối thay đổi Torch/tamper/post-test. Không chạy trực tiếp trên môi trường Windows của bạn.
