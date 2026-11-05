# Kiểm tra kỹ thuật

- Notebook17: thực thi từ đầu đến cuối, 60 score được tính lại AP/AUC/recall/FPR/fairness khớp file gốc.
- Notebook20: thực thi từ đầu đến cuối, 15 fit hoàn thành; chạy lại tái sử dụng đủ15 fit, không huấn luyện lại.
- 8 kiểm tra số học đều đạt; max absolute error < 1,3e-13.
- Pipeline18/19 đã kiểm tra trên dữ liệu tổng hợp tách riêng: 5 fit CE và20 fit cây, chọn recipe từ validation,
  đóng băng, lưu/test score, xuất review và resume. Dự đoán nạp lại khớp score đã lưu.
- RUS kiểm tra giữ toàn bộ minority và không trùng dòng; ROS kiểm tra giữ toàn bộ dòng gốc và đúng số mẫu.
- Các notebook được parse/validate định dạng; không có lỗi output ở bản17/20 bàn giao.

Kết quả dữ liệu tổng hợp dùng kiểm tra kỹ thuật, không được đưa vào bảng kết quả nghiên cứu BAF.
Runtime kiểm chứng ở đây là CPU; phiên bản thực tế nằm trong manifest và output notebook20.
25 lượt BAF bổ sung cần chạy trên cache ở máy nhóm; chưa có số liệu full-data của các lượt này.
