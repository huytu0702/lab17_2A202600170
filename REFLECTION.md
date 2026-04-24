# REFLECTION — Lab #17: Multi-Memory Agent

## 1. Memory nào giúp agent nhất?

**Long-term Redis (user profile)** giúp nhiều nhất cho trải nghiệm cá nhân hoá. Khi agent biết tên, dị ứng, và sở thích của user từ session trước, câu trả lời trở nên chính xác và tự nhiên hơn hẳn. Không cần user nhắc lại mỗi conversation.

**Semantic Chroma** giúp nhiều nhất cho factual Q&A — khi agent đã "học" được các khái niệm từ các cuộc hội thoại trước, nó có thể kết nối embedding → attention → transformer mà không cần user giải thích lại từ đầu.

---

## 2. Memory nào rủi ro nhất nếu retrieve sai?

**Long-term Redis** là nguy hiểm nhất vì:
- Lưu **PII nhạy cảm**: tên, dị ứng thức ăn, bệnh lý, địa chỉ, sở thích riêng tư.
- Nếu retrieve nhầm user_id (ví dụ lỗi collision key), agent có thể tiết lộ thông tin của người khác.
- Không có TTL mặc định → dữ liệu tồn tại vĩnh viễn dù user đã muốn "quên".

**Episodic JSON** cũng rủi ro nếu log chứa các cuộc hội thoại nhạy cảm (y tế, tài chính) và không được mã hoá.

---

## 3. Nếu user yêu cầu xóa memory, xóa ở backend nào?

Cần xóa ở **tất cả 4 backends**:

| Backend | Cách xóa |
|---------|----------|
| Short-term (MemorySaver) | `short_term.clear(thread_id)` |
| Long-term Redis | `long_term.clear_user(user_id)` — xóa toàn bộ hash |
| Episodic JSON | Filter và rewrite file, loại bỏ entries có `user_id` tương ứng |
| Semantic Chroma | `collection.delete(where={"user_id": user_id})` |

Cần implement **"Right to be Forgotten" endpoint** — một API call duy nhất kích hoạt cascade delete trên cả 4 backends.

---

## 4. Rủi ro PII / Privacy

### Dữ liệu nhạy cảm được lưu
- Tên thật, nickname
- Dị ứng thực phẩm / thuốc (y tế)
- Sở thích cá nhân (có thể suy luận về tôn giáo, chính trị)
- Lịch sử các cuộc hội thoại (episodic)

### Các rủi ro chính

1. **Key collision**: Nếu `user_id` không đủ unique (e.g. dùng username thay vì UUID), nhiều user có thể share cùng Redis key.
2. **Semantic leakage**: Chroma embeddings của user A có thể được retrieve khi user B hỏi câu tương tự (thiếu user-scoped filtering).
3. **Không có consent flow**: Hệ thống lưu mọi thông tin user cung cấp mà không hỏi "bạn có muốn tôi nhớ điều này không?".
4. **Không có TTL**: Redis facts và episodic logs không tự expire — vi phạm GDPR nếu deploy ở EU.
5. **Plain-text storage**: JSON episodic log không mã hoá — ai có quyền đọc filesystem đều xem được.

### Giải pháp đề xuất

- Thêm `consent: bool` field khi lưu memory
- TTL 30 ngày cho Redis facts (non-critical)
- Encrypt episodic JSON với user-derived key
- Filter Chroma queries theo `user_id` metadata
- Rate limit memory saves để tránh abuse

---

## 5. Limitations kỹ thuật

### Hiện tại

| Limitation | Tác động |
|------------|----------|
| Redis fallback là in-memory dict | Data mất khi restart — không dùng được production |
| Chroma disabled nếu thiếu OpenAI API key | Semantic search fail silently |
| MemorySaver in-process | Short-term không share được giữa multiple instances |
| Intent classification chỉ dùng regex | Các câu phức tạp bị classify sai backend |
| Episodic search là keyword matching | Không tìm được episode "gần nghĩa" |
| Không có deduplication | Chroma có thể chứa duplicate memories |

### Khi scale

1. **Concurrency**: MemorySaver không thread-safe khi nhiều requests cùng lúc → cần PostgresSaver.
2. **Chroma single-node**: PersistentClient không phân tán → bottleneck với nhiều users.
3. **Redis single instance**: Không có replication → single point of failure.
4. **Token budget cứng 8192**: Không adaptive theo model → phải update khi đổi sang GPT-4 (128k context).
5. **Episodic JSON file**: Race condition khi concurrent writes → cần database thật (SQLite ít nhất).

---

## 6. Lessons Learned

1. **Memory routing là phức tạp hơn tưởng** — regex đơn giản handle được 70% cases nhưng cần LLM fallback cho edge cases.
2. **Conflict handling cần explicit overwrite** — không thể chỉ append, phải có logic "latest wins" cho profile facts.
3. **Context injection phải structured** — 4 sections riêng biệt trong system prompt giúp LLM dễ tìm thông tin hơn một blob chung.
4. **Token budget management không thể bỏ qua** — nếu không trim, long conversations sẽ exceed context limit và gây lỗi.
5. **Privacy-by-design phải build từ đầu** — khó retrofit consent/TTL/deletion sau khi đã có data.
