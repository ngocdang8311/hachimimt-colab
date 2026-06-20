# HachimiMT — Dịch truyện Trung → Việt bằng AI, chạy miễn phí trên Colab / Kaggle

> **Dịch truyện tiên hiệp / huyền huyễn / web-novel Trung → Việt** bằng mô hình
> dịch máy neural (CTranslate2 INT8), cho ra văn **đọc trôi chảy như tiếng Việt
> thật** — không phải "convert" sát từng chữ. Chạy ngay trên **Google Colab** hoặc
> **Kaggle** với GPU miễn phí, hoặc cài về máy. Không cần tài khoản trả phí, không
> cần card đồ họa ở nhà.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ngocdang8311/hachimimt-colab/blob/master/HachimiMT_Colab.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/kernels/welcome?src=https://github.com/ngocdang8311/hachimimt-colab/blob/master/HachimiMT_Kaggle.ipynb)

🌐 **Bản demo online (CPU, dùng ngay không cài gì):**
[ngocdang83/HachimiMT-demo](https://huggingface.co/spaces/ngocdang83/HachimiMT-demo)

---

## HachimiMT là gì?

HachimiMT (và biến thể MoxhiMT) là các **mô hình dịch máy chuyên cho truyện mạng
Trung Quốc** — tiên hiệp, huyền huyễn, đô thị, khoa huyễn. Khác với từ điển thay
chữ, đây là mạng neural Transformer (kiến trúc Marian) được huấn luyện riêng để
**dịch nghĩa cả câu**, giữ giọng văn tiếng Việt tự nhiên, xử lý thành ngữ / điển cố
/ cú pháp Hán cổ tốt hơn cách convert truyền thống.

Mô hình rất nhỏ (35–58 MB, lượng tử INT8) nên chạy nhanh kể cả trên GPU miễn phí
của Colab/Kaggle — **một bộ truyện dài 2–3 triệu chữ Hán dịch xong trong khoảng 1
phút** trên GPU. Notebook trong repo này đóng gói sẵn toàn bộ: bấm chạy → mở link →
dán văn bản → tải bản dịch.

---

## So với QuickTranslator + Vietphrase

QuickTranslator (QT) cùng bộ từ điển Vietphrase là cách đọc truyện Trung phổ biến
nhất nhiều năm nay. HachimiMT **không thay thế hoàn toàn** QT — mỗi bên mạnh ở chỗ
khác nhau. Chọn đúng theo nhu cầu:

| | **QuickTranslator + Vietphrase** | **HachimiMT (notebook này)** |
|---|---|---|
| **Cách dịch** | Tra từ điển, thay chữ theo bảng | Mạng neural, dịch nghĩa cả câu |
| **Văn đọc ra** | Sát từ gốc, đôi chỗ cứng / Hán-Việt thô | **Trôi chảy, tự nhiên như tiếng Việt** |
| **Thành ngữ, điển cố, cú pháp Hán cổ** | Thường dịch chữ-đối-chữ, khó hiểu | Hiểu và diễn đạt lại mượt hơn |
| **Chạy offline, tức thì** | ✅ Có, không cần mạng | ⚠️ Cần GPU (Colab/Kaggle) để nhanh; bản máy/CPU chậm hơn |
| **Nhất quán tên riêng** | ✅ **Bạn pin tên trong từ điển → cố định mãi** | ⚠️ Tên hiếm có thể dịch lệch giữa các đoạn |
| **Tùy biến từ điển** | ✅ Sửa, thêm, ghi đè thoải mái | ❌ Mô hình cố định, không sửa từ điển |
| **Tốc độ** | Tức thì (tra bảng) | ~27.000 chữ/giây (1 GPU) — rất nhanh, nhưng cần khởi động |

**Tóm lại:**
- Muốn **đọc nhanh, mượt, đỡ "lai Hán-Việt"** → HachimiMT thắng về độ trôi chảy.
- Cần **offline tuyệt đối, dịch tức thì, hoặc kiểm soát chặt tên riêng / thuật ngữ
  bằng từ điển tự pin** → QuickTranslator vẫn là lựa chọn tốt.
- Nhiều người dùng **cả hai**: HachimiMT để đọc trôi chảy, QT khi cần tra đúng một
  tên riêng cố định.

> So sánh ở đây nói về **độ trôi chảy của văn dịch**, không khẳng định HachimiMT
> "chính xác hơn" — ở khoản giữ tên riêng nhất quán, từ điển pin tay của QT vẫn có
> lợi thế (xem mục [Hạn chế](#hạn-chế-cần-biết-trước)).

---

## Bắt đầu nhanh (3 cách)

### Cách 1 — Google Colab (dễ nhất, khuyên dùng)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ngocdang8311/hachimimt-colab/blob/master/HachimiMT_Colab.ipynb)

1. Bấm nút **Open In Colab** ở trên.
2. **Bật GPU trước** (rất nên — nhanh gấp nhiều lần):
   **`Runtime → Change runtime type → T4 GPU → Save`**
   *(Colab tiếng Việt: `Thời gian chạy → Thay đổi loại thời gian chạy → T4 GPU → Lưu`)*
   ⚠️ Chọn đúng **T4 GPU** — **KHÔNG** chọn **TPU** (CTranslate2 không chạy được TPU,
   sẽ rớt về CPU chậm).
3. **`Runtime → Run all`** (Ctrl+F9) — *tiếng Việt: `Thời gian chạy → Chạy tất cả`*.
4. Đợi cài đặt xong, cell cuối in ra một **link công khai `*.gradio.live`** → bấm
   vào để mở giao diện dịch.

### Cách 2 — Kaggle (có 2× GPU T4, nhanh hơn nữa)

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/kernels/welcome?src=https://github.com/ngocdang8311/hachimimt-colab/blob/master/HachimiMT_Kaggle.ipynb)

1. Bấm **Open in Kaggle** → Kaggle tạo notebook từ repo này (cần tài khoản Kaggle).
2. Phải xác minh số điện thoại để bật GPU; sau đó:
   **Settings (bên phải) → Accelerator → `GPU T4 x2`**, và **Internet → On**.
   ⚠️ **Tránh P100** (không hỗ trợ kiểu tính INT8 của mô hình) và **TPU**.
3. **`Run All`** → mở link `*.gradio.live` ở cell cuối.

Kaggle T4×2 tận dụng **cả 2 GPU** nên nhanh hơn Colab 1 GPU khoảng 1,5×.

### Cách 3 — Cài về máy (chạy offline bằng CPU)

Tải gói chạy-máy từ Space rồi chạy bằng Python — không cần GPU, nhưng CPU chậm hơn
GPU nhiều lần (xem mục Tốc độ). Hướng dẫn chi tiết nằm trong
[bản demo HF Space](https://huggingface.co/spaces/ngocdang83/HachimiMT-demo)
(mục "📦 Cài bản local").

---

## Tốc độ (đo thật)

| Môi trường | Tốc độ (beam 1, nhanh) | Bộ truyện 2,4 triệu chữ |
|---|---|---|
| **Colab T4 (1 GPU)** | ~27.000 chữ Hán/giây | ~1 phút 30 giây |
| **Kaggle T4 × 2 (2 GPU)** | ~52.000 chữ/giây (beam 1) · ~42.000 (beam 2) | ~58 giây (beam 2) |
| **CPU (bản máy / demo)** | ~500 chữ/giây | chậm — chỉ nên dùng cho đoạn ngắn |

> **GPU nhanh hơn CPU hàng chục lần.** Vì vậy luôn **bật GPU** trên Colab/Kaggle.
> Bản demo HF Space chạy CPU dùng chung nên chỉ hợp dán thử vài đoạn; muốn dịch
> nguyên bộ thì dùng Colab/Kaggle (GPU) hoặc cài về máy.
>
> *beam* càng cao dịch càng kỹ nhưng càng chậm; mặc định để 1–2 cho nhanh.

---

## Các mô hình có sẵn

App cho chọn 4 mô hình (mặc định **HachimiMT-60**, tự tải khi cần):

| Mô hình | Cỡ | Vai trò |
|---|---|---|
| **HachimiMT-60** ⭐ | 57 MB | Mặc định — cân bằng chất lượng / tốc độ, dùng cho hầu hết truyện |
| **HachimiMT-30** | 35 MB | Nhỏ nhất, nhanh nhất — hợp máy yếu / muốn tốc độ tối đa |
| **MoxhiMT-60** | 58 MB | Biến thể cùng cỡ, giọng văn khác đôi chút để đối chiếu |
| **MoxhiMT-30** | 38 MB | Biến thể nhỏ, phương án dự phòng nhẹ |

Cứ thử vài mô hình trên cùng một đoạn rồi chọn cái hợp gu nhất — mỗi mô hình ra
giọng văn hơi khác nhau.

Repo mô hình trên Hugging Face:
[HachimiMT-60](https://huggingface.co/ngocdang83/HachimiMT-60-zh-vi) ·
[HachimiMT-30](https://huggingface.co/ngocdang83/HachimiMT-30-zh-vi) ·
[MoxhiMT-60](https://huggingface.co/DanVP/MoxhiMT-60) ·
[MoxhiMT-30](https://huggingface.co/DanVP/MoxhiMT-30)

---

## Tính năng app

- 📄 **Dán văn bản hoặc tải file `.txt`** (tự nhận mã GB18030 / Big5 / UTF-8, hỗ
  trợ cả phồn thể lẫn giản thể).
- ⚙️ **Chọn mô hình** và **beam** (đánh đổi tốc độ ↔ chất lượng).
- 🈶 **Chuẩn hóa xưng hô Hán-Việt** (tùy chọn nâng cao, thử nghiệm): chuyển 哥哥/姐姐…
  sang ca ca / tỷ tỷ… theo văn phong tiên hiệp, nhận diện bối cảnh cổ trang vs hiện
  đại. Mặc định tắt — bật khi cần.
- 💾 **Xuất bản dịch ra `.txt`** để đọc offline.
- ☁️ Trên Colab/Kaggle tự tạo **link công khai** chia sẻ được tạm thời.

---

## Hạn chế (cần biết trước)

Để công bằng, đây là những chỗ HachimiMT **chưa bằng** cách dịch cũ:

- **Tên riêng hiếm có thể dịch lệch giữa các đoạn.** Mô hình nhỏ nên một số tên ít
  gặp có thể ra vài biến thể Hán-Việt khác nhau trong cùng truyện. Đây chính là chỗ
  **từ điển pin tay của QuickTranslator vẫn nhỉnh hơn** — bạn cố định một tên là cố
  định mãi. (HachimiMT không cho sửa từ điển.)
- **Cần GPU để nhanh.** CPU (bản máy / demo HF) chậm hơn nhiều lần; dịch nguyên bộ
  truyện trên CPU không thực tế.
- **Mô hình cố định**, không tùy biến thuật ngữ / không thêm từ điển riêng.
- Mô hình nhỏ → đôi khi vẫn có câu khó hiểu ở đoạn quá dài hoặc nội dung hiếm gặp.

Nếu những điểm trên là then chốt với bạn (đọc offline tuyệt đối, kiểm soát chặt
thuật ngữ), hãy dùng kèm QuickTranslator. Còn nếu ưu tiên **đọc trôi chảy, ít lai
Hán-Việt**, HachimiMT là một lựa chọn đáng thử.

---

## Liên kết

- 🌐 Demo online (CPU): https://huggingface.co/spaces/ngocdang83/HachimiMT-demo
- 🤖 Mô hình: [HachimiMT-60](https://huggingface.co/ngocdang83/HachimiMT-60-zh-vi) ·
  [HachimiMT-30](https://huggingface.co/ngocdang83/HachimiMT-30-zh-vi) ·
  [MoxhiMT-60](https://huggingface.co/DanVP/MoxhiMT-60) ·
  [MoxhiMT-30](https://huggingface.co/DanVP/MoxhiMT-30)
- 📓 Notebook: [Colab](HachimiMT_Colab.ipynb) · [Kaggle](HachimiMT_Kaggle.ipynb)

---

*HachimiMT là mô hình dịch máy neural Trung→Việt chuyên cho truyện mạng, mã nguồn
mở trên Hugging Face. Repo này chỉ chứa notebook + hướng dẫn; bản dịch do mô hình
sinh tự động và có thể cần biên tập lại trước khi xuất bản.*
