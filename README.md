# graduation_thesis

### Kế hoạch thực hiện khóa luận: https://docs.google.com/spreadsheets/d/1NTT391VXNkY8nsUlxh8JEnQkLb5So6fefUJLGQ9A8e0/edit?usp=sharing
---
### Sử dụng:

Sử dụng python3.7 để huấn luyện mô hình và phát sinh câu mô tả. Các thư viện cần thiết ở requirements.txt.

1. ***create_input_files.py***: Sau khi chạy file *create_input_files.py* các file được tạo ra lưu ở folder *data_processing_first_step* gồm có:
* WORDMAP_coco_5_cap_per_img_5_min_word_freq.json: file này là từ điển với key là word và value là index.
* TRAIN_IMAGES_coco_5_cap_per_img_5_min_word_freq.hdf5: file này chứa các ma trận ảnh  numpy có shape là *(channels, height, width)* dùng để **training**.
* TRAIN_CAPTIONS_coco_5_cap_per_img_5_min_word_freq.json: file này chứa các list caption đã được Encode từ wordmap, và đã được padding.
* TRAIN_CAPLENS_coco_5_cap_per_img_5_min_word_freq.json: file này chứa list các độ dài của các caption.
* Tương tự với các file VAL_, và các file TEST_.

2. ***train.py***: Huấn luyện mô hình và lưu mô hình tốt nhất *checkpoint/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar* (điểm BLEU-4 cao nhất).

3. ***generation_caption.py***: Load mô hình và phát sinh các câu mô tả cho tập test và được ghi file dưới dạng .json ở thư mục *evaluate/coco-caption/results/*. File này là một list các dictionary gồm 2 trường 'image_id' là id gốc của ảnh test ở bộ dữ liệu MSCOCO và 'caption' câu mô tả được phát sinh từ mô hình sau khi huấn luyện.

Sử dụng python2.7 để đánh giá mô hình. Microsoft COCO Caption Evaluation từ https://github.com/tylin/coco-caption
4. ***evaluate/coco-caption/eval.py***: Tính các điểm số của các độ đo BLEU-1, BLEU-2, BLEU-3, BLEU-4 và METEOR trên tập test.
