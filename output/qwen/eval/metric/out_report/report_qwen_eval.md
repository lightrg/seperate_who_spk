# BÁO CÁO ĐÁNH GIÁ CHÉO: QWEN 2.5 7B [BASE vs FINETUNED]

*Môi trường giả lập: RTX 4060 (Thuật toán lượng tử hóa 4-bit QLoRA Inference)*

## YẾU TỐ 1: ĐIỂM CHUẨN FORM CHÍNH XÁC (Format Adherence Score)
- **Lý thuyết:** Base model thường nói lan man, trả lời chệch hướng, không xuất đủ các mục `# 1, # 2...`. Mô hình Finetune thì được huấn luyện ép khung kỷ luật quân đội.

| Mẫu Số | Tuân thủ Format (BASE) | Tuân thủ Format (FINETUNE) | Kết Luận |
|---|---|---|---|
| 1 | 100.0% | **75.0%** | ❌ Cải lùi |
| 2 | 100.0% | **75.0%** | ❌ Cải lùi |
| 3 | 100.0% | **25.0%** | ❌ Cải lùi |

**TRUNG BÌNH FORMAT:** Base = 100.0% | Finetune = 58.3%

## YẾU TỐ 2: ĐỘ GIỐNG CÂU VĂN THỰC TẾ (Lexical Similarity %)
Đo lường tỉ lệ văn bản sinh ra có dính sát với bộ ĐÁP ÁN (Ground Truth) lúc nhồi bột hay không, ứng dụng `SequenceMatcher` module.

| Mẫu Số | Tương đồng Đáp Án (BASE) | Tương đồng Đáp Án (FINETUNE) |
|---|---|---|
| 1 | 8.2% | **7.2%** |
| 2 | 1.7% | **4.3%** |
| 3 | 17.4% | **2.6%** |

## YẾU TỐ 3: NHÌN BẰNG MẮT THƯỜNG (SO SÁNH TRỰC QUAN)

### SAMPLE DIARIZEN 1
#### 🎯 [ĐÁP ÁN GỐC TỪ CON NGƯỜI LÀM]:
```text
# 1. TỔNG QUAN CUỘC HỌP

# 2. HÀNH ĐỘNG TRIỂN KHAI (ACTION ITEMS)
Không có nhiệm vụ/deadline nào được giao trong cuộc trò chuyện này.
Chương trình "Chuyện Họ Chuyện Mình" mang đến những giây phút giải trí và thư giãn cho khán giả. Host tương tác hài hước với khách mời Hữu Đằng và Dược sĩ Tiến, xoay quanh nỗi ám ảnh trêu đùa về số 7 may mắn và những mảng miếng hài hước về "trị bệnh chính chuyên".

# 3. CHI TIẾT THEO NGƯỜI NÓI (SPEAKER INSIGHTS)
- Speaker_08 (Host): Khẳng định mục đích chương trình là mang lại sự thoải mái. Trêu đùa Hữu Đằng về những kỷ niệm "bầm mình" khi diễn chung và tương tác vui vẻ với các khách mời.
- Speaker_04 (Hữu Đằng): Giao lưu dí dỏm cùng Host, giải thích về số 7 may mắn của mình và giới thiệu sự xuất hiện của Dược sĩ Tiến để cùng lan tỏa năng lượng tích cực.
- Speaker_06 (Dược sĩ Tiến): Bày tỏ sự bỡ ngỡ đầy hài hước khi tham gia một gameshow vui nhộn nhờ lời rủ rê "trị bệnh chính chuyên" của Hữu Đằng.

# 4. KỊCH BẢN CHUẨN HOÁ (CLEANED TRANSCRIPT)
Speaker_08 (Host): Mục đích cuối cùng của chúng tôi là muốn mang cho quý vị những giây phút thật sự thoải mái. Xin trân trọng kính mời. Đằng ơi, chị sợ em quá! Chị có nói với chương trình là nếu hôm nay có Hữu Đằng thì cho chị cáo bệnh đi.
Speaker_08 (Host): Biết sao không? Tại chị sợ lắm. Lần nào đi diễn chung với em về chị cũng bị bầm mình hết. Chị thật sự sợ!
Speaker_04 (Hữu Đằng): Trời ơi, em có làm gì chị đâu. 
Speaker_08 (Host): Dạ, đây là con số của em hay sao vậy Đằng? Số bảy á hả?
Speaker_04 (Hữu Đằng): Không có, tại số 7 là con số may mắn của em.
Speaker_08 (Host): À vậy hả, chị cứ tưởng em đại diện cho số bảy luôn chứ.
Speaker_04 (Hữu Đằng): Dạ, Hữu Đằng xin gửi lời chào thân thương đến toàn thể quý vị khán giả đang xem chương trình "Chuyện Họ Chuyện Mình". Ngày hôm nay, có thêm một người bạn sẽ cùng lan tỏa những năng lượng tích cực với Hữu Đằng. Xin được mời anh Dược sĩ Tiến ạ!
Speaker_06 (Dược sĩ Tiến): Bữa nay Hữu Đằng mời anh tới đây là để anh được "trị bệnh chính chuyên".
Speaker_08 (Host): Trời ơi, cái bệnh gì mà nghe ghê vậy?
Speaker_06 (Dược sĩ Tiến): Tại vì bình thường anh không bao giờ tham gia mấy show vui nhộn thế này hết trơn, giờ vô đây không biết làm sao để hòa nhập cho vui luôn.
```

#### 🤡 [TỪ MÔ HÌNH BASE CHƯA HỌC]:
*(Nó thường hay chép lại văn bản gốc hoặc chém gió)*
```text
# 1. TỔNG QUAN CUỘC HỌP
Cuộc họp là một chương trình truyền hình mang tên "Chuyện Họ Chuyện Mình", do Đài Truyền hình Thành phố Hồ Chí Minh phối hợp cùng Madison Media thực hiện. Chương trình được phát sóng vào lúc 20h thứ 7 hàng tuần trên kênh HTV7. Cuộc họp diễn ra với mục đích chia sẻ kinh nghiệm và lời khuyên về giao tiếp, từ chối, và cách sống tốt hơn thông qua các câu chuyện của khách mời.

# 2. HÀNH ĐỘNG TRIỂN KHAI (ACTION ITEMS)
- Thảo Phạm và Quốc Anh sẽ cố gắng thay đổi hành vi của mình, đặc biệt là Quốc Anh sẽ tập luyện từ chối khi chưa đúng thời điểm.
- Thảo Phạm sẽ tiếp tục chia sẻ và giám sát Quốc Anh trong việc nói lời từ chối.
- Duy Khánh sẽ tiếp tục tham gia chương trình và chia sẻ thêm về cách sống tốt hơn.

# 3. CHI TIẾT THEO NGƯỜI NÓI (SPEAKER INSIGHTS)
- **Khả Như**: Khẳng định tầm quan trọng của việc chia sẻ khó khăn và từ chối khi cần thiết. Đề xuất việc chia sẻ về bản thân để giảm sự cả nể.
- **Duy Khánh**: Đề xuất các tình huống để giúp Quốc Anh luyện tập từ chối và chia sẻ về cách sống tốt hơn.
- **Thảo Phạm**: Đề xuất cách chia sẻ và từ chối một cách hiệu quả, đồng thời nhấn mạnh tầm quan trọng của việc chia sẻ về bản thân.
- **Quốc Anh**: Nhận ra tầm quan trọng của việc từ chối và chia sẻ về bản thân để giảm sự cả nể.

# 4. KỊCH BẢN CHUẨN HOÁ (CLEANED TRANSCRIPT)

---

Xin chào quý vị khán giả, Khả Như vui mừng gặp lại quý vị trong Chuyện Họ Chuyện Mình. Chương trình do Đài Truyền hình Thành phố Hồ Chí Minh phối hợp cùng Madison Media. Được phát sóng vào 20h thứ 7 hằng tuần trên kênh HTV7. Chúng tôi vui mừng gặp lại những khách mời và những người đồng hành.

Có nhiều người đến đây để chia sẻ câu chuyện, để cho Khả Như và những người bạn đồng hành có thêm kinh nghiệm. Điều quan trọng hơn nữa là chúng tôi muốn cho quý vị có những giây phút thật sự vui vẻ, thoải mái, nhẹ nhàng.

Ngay bây giờ chúng ta hãy cùng gặp gỡ một người bạn đồng hành, Duy Khánh.

[Bắt đầu cuộc trò chuyện giữa Duy Khánh và Khả Như]

Duy Khánh: Bây giờ tất cả mọi người xem nè các bạn. Em ơi, sao em cứ lật đật vậy? Hả? Em lật đật á. Không phải, tại vì hôm nay em biết là khi em lật đật đó nè, khi em đi thì con này nó sẽ há cái mồm ra nè. Đó, dễ thương lắm luôn á. Mà nó phù hợp với em, em mang nữa nè.

Khả Như: Ê, giống như mấy cái con mà mình chơi, mình bỏ tay vô á.

Duy Khánh: Đó. Đúng rồi. Đó. Chạy tới sáng hông?

Duy Khánh: Mệt hông em? Mời em tới đây là để tâm sự rồi.

Khả Như: Không không, chương trình gỡ bí.

Khả Như: Nói vậy qua những cái tập vừa rồi, tụi mình kiểu như làm rối khách mời, chứ không có giúp ích gì được hết.

Duy Khánh: Mà cho em hỏi là chương trình của mình là những người bạn đồng hành cùng với mèo, ngoài em ra hình như là còn các anh chị em nghệ sĩ khác nữa. Đúng rồi. Thì những anh chị em nghệ sĩ đó có đi...

Khả Như: Không có bị gỡ bí. Chỉ có riêng em với mèo thôi. Chương trình thấy em không có thay đổi nên chương trình quyết định là để cho em. Khi mà em tham gia này đi giống như bài học cuộc sống của em.

Khả Như: Ừ, để sau này khi mà em có va vấp diễn ngoài đời thì em cứ xem lại chương trình này để em hướng nghiệp.

Duy Khánh: Trời ơi, áp lực quá vậy. Nhưng mà các bạn ơi, rất là lâu rồi Duy Khánh mới có dịp được quay trở lại để gặp gỡ tất cả mọi người, cũng như là chị Khả Như. Thì Khánh về, Khánh xem lại chương trình đó, Khánh thấy mình vô tri quá. Khánh thấy nó không giống.

Duy Khánh: Mà là cái đứa trẻ bên trong của mình.

Duy Khánh: Dạ. Thế là em có một đứa trẻ bên trong hông nữa.

Duy Khánh: Có nghĩa là ở ngoài em là một người khá là hoạt bát nhưng đứa trẻ bên trong của em thì khá là nội tâm.

Khả Như: Là bên ngoài em là như một đứa trẻ trâu.

Khả Như: Còn em nói là em ở bên trong em còn đứa trẻ là còn đứa nhỏ hơn cái này.

Khả Như: Đúng rồi, là bên ngoài em là có một đứa trẻ, còn bên trong em là có một đứa trẻ nhỏ hơn.

Duy Khánh: Đúng rồi, là bên ngoài là có một đứa trẻ, còn bên trong là có một đứa trẻ nhỏ hơn.

Khả Như: Đúng, và ngay bây giờ chúng ta hãy cùng đến với chiếc hộp bí mật.

Khả Như: Thì bây giờ xin mời quý vị khán giả cùng nghỉ giải lao trong xíu xíu, chúng tôi sẽ quay trở lại ngay.

[Bắt đầu cuộc trò chuyện giữa Duy Khánh và Khả Như về chiếc hộp bí mật]

Khả Như: Quý vị thân mến, chúng ta đã cùng quay trở lại với Chuyện Họ Chuyện Mình và chủ đề ngày hôm nay là khách mời sẽ mang tới cho chúng tôi một chủ đề.

Khả Như: Mà trước khi đến với chủ đề chúng ta cùng đến với chiếc hộp bí mật.

Duy Khánh: Rồi, em nhớ chiếc hộp bí mật lắm luôn á, chiếc hộp bí mật đã mang cho em rất là nhiều những khách mời, cũng như những nguồn năng lượng đó là mới mẻ, và bên cạnh đó là những câu chuyện rất là thú vị. Và ngay hôm nay...

Duy Khánh: Ước mẹ đẹp rồi. Ơ, hảm cực bệnh màu tiên hạp hả?

Khả Như: Để coi em làm gì, tại vì chương trình cũng có gợi ý luôn.

Khả Như: Cái khúc chiếc hộp này nè. Là bữa nay đứa trẻ nào mở chiếc hộp để giới thiệu nè. Trời ơi. My God!

Duy Khánh: Một cái áo với dòng
```

#### 🏆 [TỪ MÔ HÌNH ĐÃ GẮN NÃO PEFT]:
*(Nó sẽ đóng khuôn cứng ngắc các Header #)* 
```text
# 1. TỔNG QUAN CUỘC HỌP

Cuộc họp/đoạn hội thoại được ghi âm bao gồm 35 phát ngôn từ 11 người nói. Nội dung chính bao gồm các thảo luận, chia sẻ, và thảo luận từ các phát ngôn viên. Nội dung thảo luận bao gồm các chủ đề như: Chuyện họ, chuyện mình, khách mời, chương trình, và các thảo luận khác.

# 2. HÀNH ĐỘNG TRIỂN KHAI (ACTION ITEMS)

Cần xem xét các hành động tiếp theo dựa trên nội dung thảo luận.

# 3. CHI TIẾT THEO NGƯỜI NÓI (SPEAKER INSIGHTS)

## Speaker 01
- Bây giờ tất cả mọi người xem nè các bạn. Em ơi, sao em cứ lật đật vậy? Hả? Em lật đật á. Không phải, tại vì hôm nay em biết là khi em lật đật đó nè, khi em đi thì con này nó sẽ há cái mồm ra nè. Đó, dễ thương lắm luôn á.
- Mà nó phù hợp với em, em mang nữa nè.
- Đó. Đúng rồi. Đó. Chạy tới sáng hông?
- Mệt hông em? Mời em tới đây là để tâm sự rồi.
- Mà cho em hỏi là chương trình của mình là những người bạn đồng hành cùng với mèo, ngoài em ra hình như là còn các anh chị em nghệ sĩ khác nữa. Đúng rồi. Thì những anh chị em nghệ sĩ đó có đi...
- Thì mà em sợ nha. Kiểu là em ngại nói chuyện ra. Tại vì sợ nói chuyện ra, xong mọi người cười.
- Khán giả hông có nhìn, chứ họ nói là em lại không nghiêm túc nữa.
- Bắt đầu em mới kiểu gọi là chiêm nghiệm ra là hình như vấn đề của... Là do cái cách em thể hiện nó không có tốt và cái cách nói chuyện của mình.
- Nó không có thuyết phục nên là mọi người nhầm hay sao. Cho nên những người hiểu mình thì là họ hiểu mình là mình là người như vậy. Nhưng mà những bạn không hiểu em.
- Thì họ sẽ có những cái suy nghĩ nó khác đi một chút. Hôm nay là cũng có cơ hội nên là muốn đến đây là cho chị Như và anh Khánh có thể là góp ý và chia sẻ cho em.
- Sao trong những trường hợp đó mình nói chuyện có thể là nó thuyết phục hơn, nó uy tín hơn.
- Nó về hôm nay tới đâu, và căn quan khách mật cuối cùng luôn á.
- Mèo luôn hả? Vỗ tay là mèo. Vỗ tay chứ.
- Vâng, và lời nói tiếp theo của Duy Khánh, xin lỗi là chào đến toàn thể tất cả quý vị và các khán giả của chương trình Chuyện họ, chuyện mình.
- Và ngày hôm nay, Duy Khánh cảm thấy rất vui khi mà đã thế chỗ cái vai trò host của chị Khả Như. Và ngày hôm nay, chúng ta sẽ cùng gặp gỡ một nhân vật rất là đặc biệt.
- Và đó là ai? Các bạn hãy chờ trong giây lát nhá!
- Chị Khả Như ơi, không biết là chị ngày hôm nay có háo hức để gặp khách mời của chúng ta ngày hôm nay không?

## Speaker 02
- Dạ, không phải ạ. Nó là cái áo chương trình Kim đem đến cho Ngân mặc vậy.
- Ở hiện tại thì em vẫn là một người mẫu bình thường thôi nhưng mà. Người mẫu không bình thường thì sao em?
- Người mẫu không bình thường, cái này em chưa có tìm hiểu lắm. Nên chỉ biết người mẫu bình thường như tụi em là. Đi diễn, rồi đi chụp. Rồi lâu lâu được mấy anh chị mời đến đây nè, kiểu vậy.
- Em chán hết chị, em chán định về quê luôn, là em định bán... Tính bỏ nghề, hông?
- Nó bỏ nghề nặng quá. Nó kinh vậy. Thế là khoảng thời gian mà em đi làm. Xong rồi làm khoảng thời gian thì đến một lúc mà đó em. Em nhìn lại bản thân mình. Kiểu em làm bao năm rồi mà. Vẫn váy áo thôi.
- Với lại là kiểu sức khỏe của em thì nó không...
- Ví dụ như hạn chế thức đêm. Nhưng mà ở thành phố Chí Minh thì thường hôm nào thức đêm á. Nên là vì một vài vấn đề nên là em quyết định là đi học pha chế.
- Em học pha chế để em về quê em bán nước uống. Thì em cứ nghĩ mà em bán thôi, thì sau đó là kiểu công việc của em nó dần dần ít đi á chị. Một tháng nhiều khi không có show nữa.
- Thì khi mà em không có làm diễn thì em đâu làm gì đâu. Nên là em dự định là... Đi học thêm để mình làm thêm một cái gì đó.
- Giống như là nghề tay trái tay phải đó chị. Gọi là kiếm thêm thu nhập.
- Nói xong một khoảng thời gian thì do một vài vấn đề nên là em quyết định về quê em mở ra bán. Thì sau đó thì...
- Dần dần nếu mà không ai mà kiếm em nữa thì em sẽ buông luôn. Cái gọi là kiểu đam mê của mình luôn á.
- Tại giờ trước đó là em đã mê lắm, nhưng mà sau đó từ cái đam mê của mình em biến thành cái nghề nghiệp luôn, là nghề người mẫu. Sau đó cái ngày này, em cảm thấy nó bấp bênh.
- Tại vì em nhớ là lúc em dọn rồi là một người bạn em đưa em ra. Xong lúc đó kiểu bị gượng gượng á chị.
- Đêm đi mình ngồi bán á chị, xong kiểu. Cũng hơi tủi thân một chút xíu.
- Kiểu mọi người kêu quá trời. Xong lúc đó em kiểu em.
- Kiểu nó không có kìm được nước mắt mình á. Mặc dù là em đang diễn nhưng mà.
- Em đi không nổi luôn á.
- Kiểu mọi người kêu quá, mọi người ủng hộ quá. Là em cảm nhận được tình cảm mọi người dành cho em. Kiểu em cũng chỉ là một người mẫu bình thường thôi. Nhưng mà tại sao mọi người có thể nhớ và dành tình cảm cho em nhiều.
- Em kiểu cảm thấy có lỗi, mọi người còn mong đợi mình nhiều như vậy mà mình. Dần dần mình rụt luôn. Em có ấn tượng với một hôm đó mà bạn khán giả viết cho em là.
- Thật sự là rất thích người mẫu Cao Ngân, bởi vì hay xem em lại chim dim dim. Tìm bán chàm ớt đó chị, bạn nói là Cao Ngân có chia sẻ là
```

---
### SAMPLE DIARIZEN 2
#### 🎯 [ĐÁP ÁN GỐC TỪ CON NGƯỜI LÀM]:
```text
# 1. TỔNG QUAN CUỘC HỌP

# 2. HÀNH ĐỘNG TRIỂN KHAI (ACTION ITEMS)
Không có nhiệm vụ/deadline nào được giao trong cuộc trò chuyện này.
Cuộc họp trao đổi thông tin và thảo luận công việc giữa các thành viên liên quan.

# 3. CHI TIẾT THEO NGƯỜI NÓI (SPEAKER INSIGHTS)
- speaker_08: Tham gia trình bày và trao đổi nội dung công việc.
- speaker_06: Tham gia trình bày và trao đổi nội dung công việc.
- speaker_04: Tham gia trình bày và trao đổi nội dung công việc.
- speaker_03: Tham gia trình bày và trao đổi nội dung công việc.
- speaker_05: Tham gia trình bày và trao đổi nội dung công việc.

# 4. KỊCH BẢN CHUẨN HOÁ (CLEANED TRANSCRIPT)
speaker_08: Khả Như, xin chào quý vị khán giả đã đến với chương trình Chuyện Họ Chuyện Mình. Quý vị thân mến, chương trình do Đài Truyền hình Thành phố Hồ Chí Minh phối hợp cùng với Madison Media Group thực hiện. Được phát sóng vào lúc 20 giờ thứ Bảy hằng tuần trên kênh HTV7.
speaker_08: Như thường lệ, Như sẽ đồng hành cùng với một câu chuyện của mình. Của mình và những khách mời đến đây, họ sẽ có những vấn đề mà mang đến cho Như cũng như là mong là quý vị khán giả có thêm những câu chuyện cho đời sống của mình thêm thú vị.
speaker_08: Mục đích cuối cùng của chúng tôi đó là muốn mang cho quý vị những giây phút thật sự thoải mái. Xin trân trọng kính mời.
speaker_08: Đằng đó chị sợ quá Đằng. chị sợ quá
speaker_08: Đằng, tại vì chị có nói với chương trình là nếu như mà có Hữu Đằng thì bữa đó cho chị bệnh.
speaker_08: Sao vậy? Tại chị sợ lắm. Tại vì đi diễn với em về lần nào chị cũng bầm mình hết á. Chị bị sợ lắm.
speaker_08: Trời ơi. Em có làm gì đâu.
speaker_08: Dạ. Là cái này là cái số của em hay sao vậy Đằng? Số bảy... Ờ?
speaker_04: Không có, tại số 7 là số hên của em.
speaker_08: Ai vậy hả? Chứ tưởng là em là đại diện cho số bảy.
speaker_04: Dạ Hữu Đằng xin gửi chào thân thương toàn thể quý vị khán giả đang xem chương trình Chuyện Họ Chuyện Mình.
speaker_04: Dạ, dạ ngày hôm nay, dạ có thêm một người bạn sẽ cùng lan tỏa những năng lượng tích cực đó cùng với Hữu Đằng. Dạ, xin được mời anh Dược sĩ Tiến ạ.
speaker_06: Bữa nay Hữu Đằng mời anh tới đây để cho anh được trị bệnh chính chuyên.
speaker_08: Trời ơi cái bệnh gì mà nghe ghê nữa.
speaker_06: Bình thường không bao giờ mà tham gia một cái show vui vậy trơn, giờ vô đây không biết làm sao để vui luôn.
speaker_08: Còn khi mà lên sóng là tất cả mọi người kiểu như đi họp gì đấy.
speaker_08: Có khi là lên sóng thì chỉ còn có em với anh thôi, chứ không có Hữu Đằng luôn mà.
speaker_06: Dạ đúng rồi. Thôi, thôi biên tập dựng vậy đi thôi.
speaker_08: Nhưng mà thực sự khi mà người ta mời cả ba chúng ta cùng ngồi đây thì chắc chắn khách mời ngày hôm nay phải có vấn đề.
speaker_04: Love. Chính chuyện còn hơn anh.
speaker_08: Không phải chính chuyên đâu mà em nghĩ là mình nói không nổi.
speaker_04: Là một vấn đề lớn hay sao chị?
speaker_08: Mà cả mọi người tới đây đều có vấn đề lớn.
speaker_04: Em luôn luôn muốn gặp những nhân vật đó quá.
speaker_08: Đúng rồi, nhân vật trong chiếc hộp này.
speaker_08: Em lại chiếc hộp đi, em lại chiếc hộp đi.
speaker_04: Ê, có khi nào Như lại hù em không?
speaker_08: Rồi chị đếm 1, 2, 3 nha.
speaker_08: Quý vị thân mến, trước khi đến với chiếc hộp này thì chúng ta cùng nghỉ giải lao trong ít phút. Chúng tôi sẽ quay trở lại ngay.
speaker_08: Quý vị thân mến, chúng ta cũng quay trở lại với Chuyện Họ, Chuyện Mình. Và khách mời ngày hôm nay có thêm một khách mời đặc biệt nữa, Dược sĩ Tiến. Xin chào anh ạ.
speaker_06: Giờ mình gặp lại từ đầu nữa hả? Phải.
speaker_08: Thôi thì Đằng với anh Tiến cùng mở cái này ra coi mình trông thấy là gì anh ha.
speaker_04: Trời ơi, sao chữ không? Không có hình nha.
speaker_08: Dạ em xin phép được đọc tâm thư cũng như lời gợi ý.
speaker_08: Tôi là một người phụ nữ có chất giọng đàn ông.
speaker_06: Trời ơi, chắc giọng khàn. Chắc giọng khàn, chắc bữa nay giọng khàn tới.
speaker_08: Phải không anh? Chứ em tưởng nói Hữu Đằng không đó?
speaker_08: Là con gái thì ai cũng mong mình có được một cái giọng nói dịu dàng, thanh thoát, nữ tính. Nhưng tôi thì hoàn toàn ngược lại, từ khi sinh ra, chất giọng đã khàn khàn, ồm ồm.
speaker_04: Ôi chất giọng mà như con trai ha.
speaker_08: Ôi, nghe nó gồ ghề lắm ha.
speaker_08: Anh có suy nghĩ gồ ghề là như thế nào? Anh có thấy là cái giọng nói gồ ghề không?
speaker_06: Nãy giờ là anh còn gồ ghề cái khúc đầu là không biết có dựng không á.
speaker_06: Là anh chưa có vô được cái bức thư nữa thôi.
speaker_08: Từ từ không sao đâu, em dẫn anh vô. Không thôi thì Hữu Đằng dẫn anh vô, còn không thôi hai đứa em đi ra với anh.
speaker_08: Gồ ghề và mạnh mẽ như một nam ca sĩ nhạc rock.
speaker_04: Ồ, vậy là nhìn ngầu lắm á.
speaker_08: Hay chứ bộ, về kiểu mà giọng khàn khàn như chị Hồng Ngọc đó em thấy cũng hay mà.
speaker_04: Nhanh lên! Ôi, gan dạ bình thường.
speaker_08: Nếu chỉ có âm thanh mà không nhìn hình ảnh, ví dụ như nghe điện thoại chẳng hạn, thì mười trên mười một người gọi tôi là anh.
speaker_08: Hi vọng hôm nay khi đến với chương trình, tôi sẽ được đối xử yêu thương và dịu dàng đúng như một cô gái đích thực.
speaker_06: Bây giờ nãy giờ có một mình Đằng nó cười.
speaker_04: Không, cái này không phải là cười trên nỗi đau người ta, mà cái này là cười chia sẻ.
speaker_08: Rồi, vậy thì bây giờ em đi ra ngoài, em đóng cái cửa đó vô được không?
speaker_08: Rồi đóng đi, cửa đóng đi.
speaker_04: Đi ra, mời bạn vô ạ.
speaker_04: Rồi, mừng nha, xin mời ạ.
speaker_03: Em gọi là chào mừng quý vị, quý vị và các bạn nhé!
speaker_03: Dạ, em chào mọi người và em chào chị Khả Như, anh Tiến và anh Hữu Đằng. Em là Phương Oanh. Thì em là một người sáng tạo nội dung trên nền tảng mạng xã hội.
speaker_03: Và em làm về ngành review ẩm thực.
speaker_04: Công nhận nha, cái giọng nó đặc biệt thiệt lắm. Nhưng mà riêng bản thân của Đằng nha. Đằng rất thích con gái mà có giọng giống như là Phương Thanh gì đó.
speaker_03: Dạ, thì giống như anh đang nói là... anh đang thích đúng gu, nhưng mà em không thích.
speaker_03: Tại vì em không biết hát. Em có nhiều anh chị nói em là giọng giống chị Hồ Ngọc Hà rồi đó. Nhưng mà em thấy em là trái ngược hoàn toàn của em ấy, chị. Có nghĩa là em không biết hát.
speaker_03: Là em không cảm nhận được âm nhạc luôn từ nhỏ tới lớn.
speaker_03: Thử đi, thử đi. Ý là em không biết hát luôn, bây giờ cho em một câu và chị nghe em hát đi.
speaker_03: Nếu đã xem nhau như cơ.
speaker_08: Các quốc đây, yêu bình yên thôi, yêu mãi không dây.
speaker_03: Kiểu là xem nhau như cái.
speaker_03: Thì vậy, chứ em không có lên giọng được. Bài đơn giản.
speaker_08: Bình minh ơi dậy chưa? Cà phê sáng...
speaker_08: Là lúc mà chào đời, lúc mà mới lọt lòng là một đứa trẻ nó sẽ khóc oa oa oa oa oa. Còn em mà...
speaker_03: Mà chị thấy cái giọng thế dễ thương mà, chứ có gì đâu em. Thật ra là tại vì em tới đây rồi á, nên là chị đã sẵn sàng tâm lý là có một con bé giọng khàn đang tới đây rồi. Chứ nếu như mà chị vừa nghe em thôi là chị.
speaker_03: Trong đầu của chị là người ta bị phân biệt âm. Thôi là em là con trai, cái giọng em vô quá cho...
speaker_06: Thì là hết nhẹ nhàng. Nhưng mà vậy nè, em có người theo dõi thì mấy bạn đó có thích em không? Và có vì cái giọng nói này của em mà người ta không thích em không?
speaker_03: Ừ, vậy có. Là hiện tại bây giờ em có thể là vì giọng nói này nên em là một người sáng tạo nội dung. Thì có những người rất là không thích giọng nói của em và họ nói là em cố...
speaker_03: Để gằn cái giọng trầm xuống để cho người ta nghe, người ta chú ý.
speaker_08: Thì em kể cho chị với anh Tiến cũng như Đằng nghe, quý vị khán giả nghe một số kỷ niệm cũng như là những câu chuyện của em mà em gặp phải khi mà em đang...
speaker_03: Thứ nhất là, ở lúc đầu em live stream, em bị tự ti về giọng nói của em từ trước rồi. Tức là em bị rất là không thích nói chuyện luôn. Mọi người người ta hỏi em làm sao, không trả lời luôn.
speaker_03: Và em sẽ cố gắng hạn chế nhìn vào ánh mắt của người khác là... Em bị tự ti đến cái mức như vậy.
speaker_03: Thì khi mà mình live stream thì em cứ nói thôi. Bắt đầu thì em không dám nói, nhưng mà mình... Về sau thì mình cứ ngồi và nhìn cái máy thì mình cứ nói liên thiên, liên thiên. Xong rồi cái bình luận mà khi mà người ta vừa nghe tới giọng nói của mình, người ta nhảy lên. Ôi!
speaker_03: Ui, sao giọng nói như đàn ông vậy?
speaker_03: Thật ra thời gian đầu là em hơi bị sốc với những cái comment đó.
speaker_03: Thì em cũng biết rồi, tại vì bình thường mấy anh shipper gọi cho em là không anh nào kêu em bằng em hết. Mặc dù...
speaker_03: Mặc dù là có cái anh đó giao hàng ruột cho em luôn, là gọi là mối ruột luôn rồi. Anh nói với em một câu này nè: Bé ơi, anh biết em là con gái, nhưng mà mỗi lần mà sáng anh gọi cho em giao hàng là anh vẫn cảm giác em là con trai.
speaker_03: Cái giọng của em nó không thể nào là con gái được.
speaker_08: Rồi, vậy thì những lúc mà bạn live mà người ta nói như vậy, bạn cảm thấy như thế nào?
speaker_03: Em kể cho bạn bè em nghe.
speaker_03: Em nói là thôi, em không livestream nữa đâu.
speaker_03: Thật ra là con gái tụi em nó rất là mong manh.
speaker_03: Giống như chị Khả Như cũng biết mà đúng không? Chị Khả Như giỡn vậy thôi chứ nếu như con gái của mình mà bị body shaming, bị chê ở một cái gì mình rồi đó đúng không?
speaker_08: Khóc đó. Trời ơi, một trăm hòn đá mà ném vào đâu đó thì cũng có một, hai viên trúng mình.
speaker_08: Như lại thấy em hay đó. Tại vì đối với cái nghề của Như, cái giọng nói trong hay là trầm, nó không quan trọng bằng... Nhưng mà giọng trầm nó hay hơn.
speaker_04: Thật sự nãy giờ là Đằng nhìn Phương Oanh là Đằng không chớp mắt luôn. Có chớp mắt đôi lần. Tại vì nó cay quá.
speaker_04: Nhưng mà tức là sao, tức là gây được sự chú ý đối với Đằng. Một cái sức hút gì đó, rất nhiều đối với Đằng. Mà lúc đầu Đằng cũng nói là Đằng rất thích những người phụ nữ mà có cái giọng trầm vầy. Bây giờ trong mắt Đằng, những người phụ nữ đó là những người.
speaker_04: Thứ nhất là cá tính. Cái thứ 2 là gợi cảm. Mình thấy ở họ là một con người rất là thẳng thắn với nhau. Có gì cứ nói thẳng, chia sẻ thẳng với nhau. Và họ rất là mạnh mẽ.
speaker_03: Có một cái chuyện đó là... Em lúc mà mới quen người yêu của em. Thế là kiểu như là nếu mình quen người yêu đi thì mình sẽ nói chuyện bằng video call. Thì khi mà người yêu nói chuyện với em á.
speaker_03: Ở trên video call nữa thì...
speaker_03: Gọi bạn đó vô, xong rồi khóc. À ờ.
speaker_08: Bởi lắm, dạ, cái thanh quản trời.
speaker_04: Tưởng con mình không đi thẳng mà.
speaker_06: Nhưng mà rồi sao đó bạn người yêu em giải thích cho mẹ xong thì mẹ có hiểu không?
speaker_06: Ừ, dễ thêm câu chuyện làm dâu, dễ mà con dâu có ấn tượng trước với mẹ chồng luôn. Có kỷ niệm vui luôn chứ. Còn nếu mà giọng em bình thường nhiều khi mẹ không thèm hỏi tới, không nhớ em là ai. Sau này mà lỡ bạn trai em còn có đổi bồ thì cũng không hay luôn.
speaker_08: Đúng rồi, em thấy. Em thấy cái này hay nha. Khó lừa à.
speaker_08: Bây giờ bạn trai của bạn có thói quen video call. Thì lỡ một ngày nào đó ảnh nói chuyện với nhỏ khác. Mẹ của bạn trai em sẽ gọi điện méc em. Nói cho em nha.
speaker_08: Có một con mắm nào đó xuất hiện trong cuộc đời ảnh rồi.
speaker_06: Nhưng mà giờ cái vấn đề của em là. Bây giờ em nói là em tới đây chia sẻ bởi vì em vượt qua được nó rồi. Thì em hồi đó em làm cách nào mà em vượt qua được cái thời mà đau đớn, khủng khiếp như anh hồi xưa, anh bị người ta chửi nha. Làm kiểu là anh muốn bỏ Sài Gòn về quê luôn, anh không muốn trên đây nữa.
speaker_06: Chứ không nói, thì em làm sao? Tại vì anh nghĩ là... Mấy bạn nữ mà cũng tương tự em á. Mấy bạn cũng muốn biết cái cách làm sao để vượt qua được nó. Chứ giờ tụi anh mà khuyên nó. Ừ, không sao đâu, không sao đâu. Mấy chuyện này bình thường khó vô lắm. Người ta sẽ nói ừ.
speaker_06: Ông Tiến nè, bà Như nè, ông Đằng nè. Thành công rồi cho nên nói gì cũng được hết á. Mấy bạn muốn... Nghe câu chuyện của em thì sao?
speaker_03: Em sẽ bật mí cho mấy anh chị muốn biết là một sự thật về em là. Bản thân em là sống trong việc chế giễu ngoại hình. Chắc là 20 năm.
speaker_03: Tức là trước đây em là cô bé, nặng 105kg.
speaker_03: Dạ dạ em hiện tại là 55 ký.
speaker_06: Chắc là phải xin một số duyên chia sẻ bí quyết giảm cân luôn.
speaker_08: Cho tui xin với, tui cần giảm 3kg đâu mà tui còn giảm chưa được nữa nè. Trời ơi.
speaker_06: Vì vậy nó khá là động lực lắm luôn.
speaker_03: Vậy là lúc mà em đi khám, em gặp bác sĩ về chế độ dinh dưỡng, nói chung là bị béo phì. Đầu tiên luôn là năm em học lớp 3, là lúc đó em còn nhớ tại thời điểm này em đã béo phì cấp độ 3 rồi.
speaker_03: Và em vô nghe rất là nhiều thứ về bác sĩ chia sẻ cho em luôn. Và cũng khóc, nhưng mà cuối cùng rồi cũng ăn. Ờ.
speaker_03: Mà em lớn đến năm 20 tuổi. Thì đỉnh điểm của năm 20 tuổi của em là 105 cân. Nói chung là trong cái quá trình dài vậy thì cái việc mà chế giễu nè. Rồi là miệt thị nè.
speaker_03: Nó chồng chất lên trên em. Thì nó thành ra một cái tính cách của bản thân em bây giờ. Giống như là khi mà bản thân em dù là. Hiện tại như chị chị nếu nói là. Em là một... Người đại diện cho người tiêu dùng. Em rất là lanh trên rất là nhiều người.
speaker_03: Nhưng mà đôi khi là sâu bên trong con người em vẫn có 1 cái. Nỗi sợ. Vẫn có 1 cái sự tự ti rất là nhiều, trước đám đông.
speaker_03: Đây em xin chia sẻ một câu chuyện đó là nó hơi buồn.
speaker_03: Là hồi cấp 2 là lớp 9. Lúc đó thì con trai rất là quậy trong lớp. Mấy bạn giỡn với nhau. Thì mấy bạn khi một bạn... Vô tình làm cái hành động là trọi cái thùng rác. Nhưng mà các bạn trọi là trúng người em.
speaker_03: Thì thay vì nhận được cái lời xin lỗi. Thì em lại nhận được câu là. Ai biểu mày mập quá làm chi. Mày bự con thì cái sọt rác dính mày lại đúng rồi.
speaker_03: Nó là một câu chuyện mà, chắc là đến tận bây giờ luôn nha, là em không có quên được câu chuyện đó. Em không quên được những hình ảnh ánh mắt của những người trong lớp mà nhìn mình.
speaker_03: Giống như là tự nhiên mình là một người bị người ta tác động vô, mà mình lại... Tự nhiên bị thêm một cái lời miệt thị nặng như vậy nữa.
speaker_04: Và những cái điều đó nó chất chồng. Thành thử ra nó thành một cái động lực. Để em đủ cái tâm em phải giảm cân.
speaker_08: Bây giờ nữa hả? Tự dưng giây phút này lắng đọng quá. Quý vị khán giả, chúng ta cùng nghỉ giải lao trong ít phút. Chúng tôi lắng đọng chút xíu rồi. Đừng đi đâu cả, chúng ta sẽ quay trở lại ngay.
speaker_08: Mình sẽ cùng chơi một trò chơi đó là mình đoán calo của những cái thức ăn. Mình ăn.
speaker_03: Dạ, em tự tin, tại trong khoảng thời gian mà em giảm cân thì mấy cái này là em phải học thuộc. Vậy hả? Dạ.
speaker_06: Thú thật luôn á chị. Hôm nay em đến á. Anh nghĩ không phải em cần cái giải pháp gì từ anh hay chị Khả Như hay anh Đằng đâu. Em muốn mang một thông điệp tích cực tới, thì đó cũng là điều anh muốn hỏi từ đầu, có nghĩa là ở cái giai đoạn mà cùng cực nhất của em.
speaker_06: Thì cái điều gì đã biến đổi em từ một cô bé chịu không nổi.
speaker_06: Lại có thể có động lực để vượt qua cái nhiêu đó. Cái khúc đó mới là cái khúc quan trọng nhất nè.
speaker_08: Quý vị thân mến, chúng ta cùng quay trở lại với chuyện họ, chuyện mình và câu chuyện rất truyền cảm hứng đó là bạn đã nỗ lực giảm cân có số ký lận, giảm 50 ký hả em?
speaker_06: Xỉn. 50 mấy ký trong 3 năm là vị chi là 36 tháng giảm 50 ký. Là tính ra hàng tháng giảm khoảng ký rưỡi đấy em.
speaker_08: Thật sự luôn, thấy con người ta làm ăn không, dân làm ăn không?
speaker_04: Thiệt mới có mấy phút quảng cáo mà tính ra được, bà tính hay quá trời.
speaker_08: Dạ. Trên tay mọi người là 1 cái tấm bảng, bây giờ mình sẽ cùng nhau. Tại vì cái vấn đề mà giảm cân cũng là 1 cái điều mà em thấy đó. Giống như bản thân em là dành cả thanh xuân để giảm cân luôn.
speaker_08: Ngày nào em cũng suy nghĩ về chuyện giảm cân hết. Và em thấy... Bây giờ nó là cái vấn đề mà. Nó giống như là nó đồng hành với mình luôn rồi. Nên bây giờ hôm nay mình sẽ cùng chơi một trò chơi.
speaker_08: Đó là mình đoán calo của những cái thức ăn mình ăn.
speaker_04: Vậy thì em kiểu em bị thua rồi. Sao vậy?
speaker_04: Một bên là kinh nghiệm về giảm cân rồi.
speaker_04: Là phải tìm hiểu rồi, một bên là dược sĩ luôn rồi. Và cũng phải cân đo đong đếm rồi.
speaker_06: Nói nghe nè. Bình thường là tui đang xây dựng hình tượng tui trên các chương trình là dạng như cũng thông minh, hiểu biết lắm hả. Rồi vô đây chơi cái này mà lỡ mà nói sai là coi như sập đổ hết, tội biên tập.
speaker_06: Thế còn bây giờ nói nghe là lỡ mà nói sai là không dựng được không?
speaker_08: Anh không biết hả? Có nghĩa là. Người ta thấy những cái chương trình khác. Anh kiểu mà anh rất là chặt chém. Anh rất là dữ dội đó. Thì những người đó mới hùn tiền lại. Gửi anh vô chương trình này. Dạ đúng rồi.
speaker_08: Gửi anh vô chương trình này để dập anh.
speaker_08: Để anh có được bài học của cuộc sống.
speaker_08: Nói chơi chứ thật ra. Em nghĩ là kiến thức về giảm cân, về calo. Không phải ai cũng biết. Nhưng mà mình chơi thử thôi mình hiểu được nhiều.
speaker_03: Dạ, em thú thiệt, tại trong khoảng thời gian mà em giảm cân thì mấy cái này là em phải học thuộc.
speaker_03: Dạ, học thuộc bài luôn đó chị.
speaker_08: Món đầu tiên đã là... Kinh thiệt lòng.
speaker_06: Cho anh hỏi là có tính luôn cái chén muối tiêu không em?
speaker_06: Có trái ớt trên hột vịt lộn mình có cộng vô luôn không? Bây giờ mình cá độ rồi mình phải rõ ràng chứ em.
speaker_08: Trứng lộn thôi, dược sĩ ơi.
speaker_04: Dạ, của em đó chính là...
speaker_08: Còn của bạn là 150 calo, còn dược sĩ Tiến là...
speaker_06: Tính vỏ thôi đúng không anh? Dạ không, nó mới đập ra, nó chưa có ăn, nó đập ra trơn nè. Chưa có ăn là đã có calo.
speaker_06: Nó mới đập có cái vỏ không hả?
speaker_06: Chừng nào ăn vô bụng mới có calo chứ. Vậy là anh tính cái số calo mình mất khi mình đập á. Thì đó.
speaker_08: Khúc này tôi coi như dược sĩ Tiến không phải chơi nha.
speaker_08: Tại vì cái kết quả anh ghi ra nó bị loại. Bị loại. Loại luôn.
speaker_08: Nếu như trong trường hợp mà mình ăn hết cho hột vịt lộn này. Thì mình sẽ có thêm 182 calo.
speaker_08: Rồi đó, của Đằng gần đúng nha.
speaker_08: Đây, món tiếp theo đó là chè bưởi. Nếu như mà mình ăn hết cái chén chè bưởi này thì sẽ được bao nhiêu?
speaker_06: Anh định hỏi câu nữa mà không dám hỏi. Sao ạ? Chè bưởi thì có đường không?
speaker_06: Thì nếu mà mình nói mà mình có đường, nó sẽ khác với không có đường. Chứ là mình thường ăn chẳng mức không có đường.
speaker_08: Chè bưởi ở mức độ là đường trung trung dùm em hả? Chè mà, chè bán, chè đường mà ăn đường đó dùm em.
speaker_06: Thế nhưng mà hoặc đường phèn hay cái đường khác nè.
speaker_08: Đường phèn với đường khác nữa nè. Là cái lượng đường đó giống nhau hông? Khác nhau.
speaker_04: Đúng rồi. Vậy ít hơn đường cát. Cát nha chị, cát đi.
speaker_04: Cát vàng thì cát trắng kìa.
speaker_08: Chứ sao nữa, mình mệt quá.
speaker_08: Rồi, kết quả xin mời ạ!
speaker_08: Ba trăm calo. Rồi mà em. Ba trăm năm mươi calo. Còn...
speaker_06: Anh mới ghi kịp con số em nó chưa gọi, anh chưa ghi kịp vì 2 số tiếp thêm.
speaker_08: Câu trả lời của vượt sĩ Tiến một lần nữa nha.
speaker_06: Tôi đang xây dựng hình tượng tui trên các chương trình là dạng như cũng thông minh hiểu biết lắm á.
speaker_08: Dạ chè bưởi, với số calo, đó là 428 calo. Wow. Qua quán được 1 tiếng cộng, còn vô số tiếng loại thêm câu nữa.
speaker_08: Và chúng ta cùng đến với 1 món ăn tiếp theo, 1 cái loại mà ngày Tết chúng ta hay. Mà chúng ta nghĩ là nó sẽ không mập, nó chỉ nóng thôi.
speaker_08: Hạt dưa. Thường hạt dưa mình hay ngồi mình ăn, ăn, ăn, liên tục, liên tục 100g hạt dưa. Thì sẽ tiếp thu vào trong cơ thể mình là bao nhiêu calo.
speaker_08: Ê, con số này bị bất ngờ nha.
speaker_08: Coi kìa, coi kìa. Cái ngồi xí kìa. Chơi cho đàng hoàng coi vượt sĩ.
speaker_04: Thế em đoán là như vậy. Vừa. Tại vì em có ăn luôn vỏ. Cho nên phải chi tiết để mọi người hiểu lầm.
speaker_04: Thôi hãy đưa khóa đọc đi mẹ em ạ. Đó là 285 phẩy.
speaker_04: Ê, em chết, em ghi lộn số căn cứ của em.
speaker_04: Dạ, em đoán lại, 285 thôi ạ.
speaker_08: Thôi! Nhanh xuất dạng các bố! Nhanh xuất quả! Hạt dưa nếu như mà chúng ta ăn hết 100 gram thì chúng ta sẽ...
speaker_08: Và con số đến là nhanh 557 calo.
speaker_08: Anh ơi, ba ngàn là có như là cái số tiền mình mua hạt dưa đó hả? 400 ngàn là mình mua nè. 285 là tiền thiếu nợ của em hay sao hả bé ơi?
speaker_06: Ê, ý là em là em sai. Hồi nãy em gợi ý em sai, em nói con số này nó lạ lắm. Con số 500 nãy anh đoán rồi. Em nói lạ lắm rồi mới đoán tới 3.000.
speaker_08: Ý là... Ý là... Ý em nói lạ là có nghĩa là em không nghĩ hạt dưa nó nhiều calo vậy.
speaker_08: Dạ rồi. Vậy thì kết quả ngày hôm nay ai là người chiến thắng, bạn quên?
speaker_04: Xin cảm ơn bạn đã đến với chương trình của chúng tôi trong ngày hôm nay. Và bây giờ đây là phần thưởng của bạn.
speaker_04: Đó chính là một bữa ăn rất là thịnh soạn.
speaker_08: Ok, bây giờ chúng ta sẽ cùng đến 1 phần thưởng của bạn. Mà cũng giống như là phần thưởng cho chúng tôi là chúng tôi sẽ được ngồi xem trực tiếp bạn review 1 món ăn gì đó.
speaker_03: Em sẽ review giống như là trên clip em hay làm nha.
speaker_03: Là thường những cái món ăn này em chưa ăn liền, em chưa bỏ vô họng mới mang. Mà tụi em sẽ cầm tụi em dơ lên, tụi em sẽ xới lên coi trong đây có cái gì.
speaker_03: Đó, mình sẽ coi thử trong đây có cái gì. Thì trong này sẽ có trứng cút nè, khô bò, khô gà nè. Đó, rồi bánh tráng.
speaker_03: Rồi bắt đầu mình sẽ lấy 1 miếng trong đây đầy đủ tất cả các món luôn.
speaker_03: Mà mình bốc lên một miếng to. Tại mình sẽ há cái họng mình. Tại vì bánh tráng trộn nó mình phải ăn.
speaker_03: Nhưng mọi người biết mấy cái độc lạ phải ăn to hơn nhiều.
speaker_03: Mà để không dính son thì em phải lấy cho nó gọn lại.
speaker_03: Chị ơi! Chị đừng có giỡn vậy mà chửi. Nè! Cũng này là em nè. Rồi, em ăn nha mọi người.
speaker_03: Dạ không, đồ ăn là tùy khẩu vị của mọi người. Em thấy ngon, nhưng mà anh Nụ Đàng không thích ăn bánh tráng.
speaker_03: Thì anh Nụ Đàng mới không ngon. Nhưng mà đối với cái miệng này em thích nào ăn vặt nên em thấy.
speaker_08: Vụ đó thì, không như mà bạn ăn nhìn thấy ngon, cái cách gắp của bạn nhìn ngon luôn.
speaker_08: Ờ, cái kích thích vị giác ơi, cũng như là cái nhìn thị giác luôn. Nhìn thấy ngon lắm, cảm ơn em rất nhiều luôn. Xin quý vị cùng giải lao cho mấy phút, chúng tôi sẽ quay trở lại ngay ạ.
speaker_08: Xin quý vị thân mến, chúng ta cũng sẽ trở lại với Chuyện Họ Chuyện Mình. Em ơi, câu chuyện em mang tới, nó đầy đủ hương vị hết. Nhưng mà cái vấn đề của em thật sự khi mà em mang đến chương trình đó là gì?
speaker_03: Ừm, em muốn mang đến chương trình là ở ngoài kia cũng có rất nhiều bạn giống như em luôn, là bị chê giễu về ngoại hình của mình.
speaker_03: Tức là mọi người sẽ không biết được là việc chê giễu ngoại hình nó ảnh hưởng rất nhiều đến tâm lý của một người, nhất là những bạn gái. Và ở độ tuổi mà mới lớn thì thường là nó bị thu thập.
speaker_03: Vô rất là nhiều và hình thành ra một tính cách của một người luôn. Ví dụ với bản thân cá nhân của em đi. Khi mà em mập là khi béo đi. Thì lúc đó là vừa con gái, vừa đen.
speaker_03: Rồi đương nhiên béo quá thì mấy bạn nam hầu như đều chọc hết. Và luôn luôn là tâm điểm chú ý.
speaker_03: Để cho người ta nhìn vào mình và người ta đánh giá rồi. Nhìn em thì mọi người sẽ rất là thú vị. Em đen rất là xướng. Nhưng mà xung quanh mọi người em lại không biết là có một giai đoạn là.
speaker_03: 20 năm, em sống với cái ngoại hình và em bị tự ti. Và khi mà mình lớn lên rồi, mình bắt đầu gặp nhiều người hơn. Giống như em nói chuyện chị Như đi, thì em lại bị tự ti, bị thu, mình lại.
speaker_03: Thì lúc nào mình cũng vẫn có 1 cái khoảng lặng ở bên trong mình. Và mình sợ là sự chú ý... Và tập trung vào mình. Nó thành ra một tính cách của mình luôn, khi mà mình bị người ta chê giễu về ngoại hình của mình.
speaker_06: Chuyện của em vậy nè, anh thấy là...
speaker_06: Có thể là mình đã có một thời gian dài, mình phải trải qua cái việc là chống chọi với những lời nói bên ngoài. Nhưng mà ít ra cho thấy thời điểm hiện tại thì em tương đối hài lòng với những gì em có rồi.
speaker_06: Vậy thì anh hỏi ngược là thí dụ như hồi đó mình nhiều ký như vậy. Mà không ai nói với mình hết. Thì ngày hôm nay mình có được như vậy không?
speaker_06: Xin chào, tạm biệt, tạm biệt. Tất cả tụi mình ở trên đời, khi mà được trao tặng cho một cái gì đó, nó khác biệt.
speaker_06: Nó có thể là một cái gì đó để người ta nhìn vào mình. Người ta nói này nói nọ. Nhưng mà nếu mình biết cách á, thì nó sẽ là một món quà.
speaker_06: Ở chỗ của em, hiện nay em có được cái động lực để mà em siêng tập thể dục thể thao. Em siêng vận động để em giữ dáng. Là nhờ 20 năm em sống chật vật với những cái điều người ta nói cho em.
speaker_06: Còn anh bây giờ tập 2 ngày thôi là anh thấy mệt, anh không có động lực tập rồi. Chị Khả Như mà phải giảm trong cả cuộc đời là bởi vì chị cũng không có động lực tiếp tục tập.
speaker_06: Thì nếu mà một ngày mà Khả Như với Tiến bị chê thậm tệ, thì lúc đó cũng sẽ có động lực thôi. Vậy thì ở đây mình đang muốn nói cái vấn đề tích cực hay tiêu cực của việc người ta chê mình. Bởi vì những cái ở bên ngoài là những cái mình không điều khiển được.
speaker_06: Cái duy nhất mình có thể thay đổi được, cái ở bên trong.
speaker_06: Vậy thì em đã làm được một điều thành công, đó là em sống và vượt qua được những cái điều đó. Còn anh biết ngoài kia sẽ có nhiều bạn là bị những cái lời nói đó và có thể là họ sẽ ngại luôn.
speaker_06: Họ không vượt qua được và thậm chí là đã có những chuyện...
speaker_06: Buồn xảy ra với một số bạn mà không vượt qua được điều đó.
speaker_06: Vậy thì hôm nay em đến á. Anh nghĩ không phải em cần cái giải pháp gì từ anh hay chị Khả Như hay Đăng đâu. Em muốn mang một thông điệp tích cực tới, thì đó cũng là điều anh muốn hỏi từ đầu, có nghĩa là ở cái giai đoạn mà cùng cực nhất của em.
speaker_06: Thì cái điều gì đã biến đổi em từ một cô bé chịu không nổi.
speaker_06: Lại có thể có động lực để vượt qua cái điều đó. Cái khúc đó mới là cái khúc quan trọng nhất nè.
speaker_03: Lúc đó là em có một người bạn, bạn rất là chăm chút ngoại hình. Xong mà nó mới nói với em vậy nè. Cái điều mà em thích là gì và cái điều mà em muốn làm là gì. Nếu như mà em thay đổi cái ngoại hình của em. Em hãy đứng nhìn trước gương á.
speaker_03: Nhìn vô bản thân của mình, nếu như mà em thích bản thân em tại thời điểm đó thì em đừng thay đổi. Còn nếu như bản thân em đứng tại thời điểm đó mà em ghét bản thân mình, thì hãy giảm những cái gì mà mình đang ghét trên người mình xuống.
speaker_03: Thì lúc đó em nghe xong em mới về làm và đúng. Đúng là khi mà em nhìn trước gương, em rất là muốn được giống như các bạn. Được mặc đồ đẹp, được có size quần áo bằng với các bạn.
speaker_03: Em chia sẻ mọi người luôn là giống như mọi người nói là mọi người tập 2 ngày ra phòng gym mọi người chán rồi. Là thật ra em là. Em phải sống như là cái việc mà chán đó. Trong một thời gian dài luôn. Là em chán nhưng mà em phải đi, em chán là em phải đi.
speaker_03: Thì bắt đầu thời gian đầu là em chỉ tập ở nhà thôi vì em thực sự là em bị tự ti. Vì mà ngoại hình của mình em không dám đi ra phòng tập, đi ra phòng tập, lỡ người ta nhìn mình là em sợ. Thì ở đó ai cũng đẹp hết trơn, còn mình, mình xấu mà mình không dám ra phòng tập.
speaker_03: Sau đó thì bắt đầu thời điểm đó em xuống được tầm 20 cân là tập ở nhà rồi em chỉ ăn uống, ăn rau thôi. Và em cắt, thực ra là lúc đó em bị cắt tinh bột luôn.
speaker_03: Để em giảm cân vào lúc đó, thật ra bao nhiêu cái công thức về giảm cân em áp dụng hết luôn. Em cứ áp dụng liên tục từ công thức này qua chế độ nọ. Liên tục, liên tục, liên tục cho một khoảng đầy chán dài luôn.
speaker_03: Thì bắt đầu về sau này, em mới tự định hình cho mình được là mình tập như thế nào mới là hợp với mình. Rồi mình phải ăn như thế nào mới là hợp với mình.
speaker_03: Bắt đầu em dần dần để thích nghi được với cơ thể của mình và thật ra lúc mà em giảm được cân. Và trong quá trình giảm cân là mình phải thực sự yêu cơ thể của mình.
speaker_03: Còn nếu như mà mình càng ghét cơ thể của mình là mình càng khó giảm cân.
speaker_04: Còn bản thân em thì giả sử là em đặt bản thân mình không phải là nghệ sĩ. Không phải là người của công chúng, làm như một người bình thường đi. Thì về cái việc cân nặng đó thì nếu mà là em, là em không quan tâm.
speaker_04: Em thực sự không quan tâm điều đó, mà em quan tâm cái tinh thần. Cái tinh thần của người đó lúc nào cũng phải vui vẻ là quan trọng.
speaker_04: Mình tỏa ra một cái năng lượng mà người tiếp xúc với mình cảm thấy dễ chịu, cảm thấy thoải mái là được rồi.
speaker_08: Đúng rồi, đúng rồi. Cái đó quan trọng.
speaker_04: Và em có một cái này cũng muốn chia sẻ là nếu như các bạn ở trên mạng, mình có cầm điện thoại, mình chuẩn bị viết một câu nào để chê bai, comment bình luận cho một người khác, thì bạn vô tình đã gieo một năng lượng tiêu cực qua trong người đó.
speaker_04: Và bạn, chính bạn cũng đang nhận về một năng lượng tiêu cực. Tại vì. Lửa mà bạn đang giữ trong tay, bạn định ném cho người ta. Là trước khi bạn ném, bạn đã bị phỏng rồi.
speaker_04: Cho nên là thành sự ra chúng ta hãy chia sẻ gì, chia sẻ điều tích cực. Còn nếu như mình cảm thấy là ừ điều đó tôi không chia sẻ tích cực được. Thì thôi mình im luôn. Mình khỏi cần nói tới điều đó để làm gì.
speaker_08: Nha, cảm ơn Phương Oanh vì ngày hôm nay em đến. Em mang một câu chuyện truyền cảm hứng. Thật ra những cái vấn đề về giảm cân nó có rất là nhiều chương trình nói về điều này.
speaker_08: Nhưng mà cái mà hôm nay mà Oanh cũng như là anh Tiến với Khả Như với anh Hữu Đằng nói. Ngày hôm nay để cho mọi người thấy là. Thật ra không có cái gì mà chúng ta không vượt qua được hết.
speaker_08: Mà chúng ta phải vượt qua như thế nào bằng cái sự thoải mái nhất của mình. Người ta hay khuyên đó là hãy là chính mình. Nhưng mà với Như đó là hãy là chính mình. Một phiên bản tốt nhất nên thành ra.
speaker_08: Khi mà mọi người cảm thấy mình ở phiên bản tốt nhất. Mọi người sẽ hạnh phúc hơn. Cám ơn Oanh ngày hôm nay. Và mong những điều mà em mong muốn sẽ trở thành hiện thực. Em...
speaker_03: Xin cảm ơn chị Như, anh Tiến và anh Đăng đã cho em những lời khuyên rất là xịn luôn. Em nghĩ là sau cái chương trình này mà nhiều bạn bị chê nhiều về ngoại hình sẽ rất là...
speaker_03: Muốn nghe được những lời giống như anh Tiến nè. Chị Như nói với em. Nó rất là giúp ích cho mấy bạn. Để cảm giác là mấy bạn không bị cô đơn một mình.
speaker_08: Và có thêm động lực để làm đẹp cho bản thân mình. Mình làm đẹp không phải vì ai đâu. Mà vì bản thân mình đó. Và đây là sự lo lắng, nỗi lo lắng của cái bề ngoài của em.
speaker_08: Và nếu như em cảm thấy ngày hôm nay đã giải tỏa được cho em thì. Em làm bể nó đi. Giống như một lời hứa với bản thân mình, mình sẽ tốt hơn mỗi ngày.
speaker_03: Cũng như là tất cả những cái bạn nào bị dè bỉu về ngoại hình. Hãy luôn tích cực đi. Tại vì. Nếu có một ngày đó các bạn gặp được những anh chị như là chị Như. Anh Tiến với anh Đăng thì bạn sẽ thấy được một cái góc nhìn tích cực hơn.
speaker_03: Của bản thân mình. Về. Cái việc mà các bạn đang bị tự ti.
speaker_08: Cảm ơn các em rất nhiều, chào tạm biệt em!
speaker_08: Dạ, mình cùng xem coi bên trong nó là gì.
speaker_06: Mày gói kỹ dữ vậy trời.
speaker_06: Hộp này thấy giống hộp vàng đó.
speaker_08: Giờ ơi tui tức lắm.
speaker_08: Trời ơi, đây là phần thưởng của em. Sao bao nhiêu ngày cực khổ cùng với tất cả mọi người em đã được thưởng? Thầy Quang đi. Cái này là người ấy ý mà.
speaker_06: Rồi sao không có tâm thư gì như mấy lần trước hả cưng?
speaker_08: Thì bởi vậy bây giờ mình nhìn hộp là mình biết người nè.
speaker_08: Đăng ơi em đi ra ngoài ngõ rồi em coi có khách mời hôm nay không, dùm chị coi.
speaker_04: Cảm ơn mọi người đã đợi cho mình cái hộp này, ta...
speaker_08: Đúng rồi, đưa cái này là phải đưa cái đồ dặn cho mình, mình biết đeo được chứ anh. Đây, đây, đây, đây, đây, chị ơi. Có, có, có. Chào anh. Dạ. Dạ, đây dạ. Dạ, chào anh. Đến với gia đình.
speaker_08: À, cái này là gợi ý chứ phải quà, không? Mình là hồi lấy lợi đúng không?
speaker_06: Và bây giờ 3 đứa mà chọc có 2 cái rồi đứa nào lấy đứa nào nhịn chứ? Đi, sao anh?
speaker_05: À, ngoài xe em còn, ngoài xe còn mấy cái anh lấy ngoài xe. Dạ em còn một thùng.
speaker_04: vàng thiệt mà đi đâu đem theo cả thùng vậy.
speaker_05: Em sẵn đi quay chương trình, em đi giao cho người ta luôn.
speaker_08: Anh chia sẻ chút xíu đi sao mà anh đánh đầy dí quá nhiều của cải như vậy.
speaker_05: Dạ, chào mọi người, em đến từ quận 5, bên công ty của em là chuyên sản xuất và phân phối bên mặt hàng là trang sức bạc và nữ trang ở bên gian.
speaker_05: Thì em cũng là một người con trong gia đình làm nghề có truyền thống. Nghề là tới đời em là ba đời rồi. Làm bên ngành kim hoàn nữ trang. Nhưng mà tới đời em thì em hết làm rồi.
speaker_05: Dạ em có làm nhà nha, đi bán nè.
speaker_08: À, đi bán là mình không làm tại nhà nữa.
speaker_05: Dạ không, ở xưởng vẫn sản xuất và vẫn phân phối sỉ, nhưng mà làm nghề á chị, có nghĩa là mình cầm, mình ngồi, mình cứ, mình cắt thế này.
speaker_05: Họ như mình thì bây giờ là em... Kinh doanh nó sẽ...
speaker_05: Em bán rồi. Đúng rồi, bán thôi. Nó sẽ có lợi nhuận hơn là ngồi làm.
speaker_08: Nhưng mà trước đây anh đã từng ngồi chạm chỗ vàng chưa?
speaker_05: Dạ có, em cũng làm bên nghề, nhưng mà kiểu như là không có chuyên bằng mấy chú.
speaker_05: Hay là ba ở nhà hay là các thợ ở nhà.
speaker_06: Gọi dạng như là ở nhà, bây giờ nhiều dạng quá là áp lực quá, lên đây chia sẻ bớt hay sao cưng?
speaker_05: Dạ em cũng không có dự định đó nhưng mà em nghe nói vậy em cũng... Để em xin nghĩ lại.
speaker_08: duy, cái gì duy ạ?
speaker_08: khánh duy còn gia bảo này là
speaker_05: Dạ cái này của con con.
speaker_05: Dạ. 4 tuổi rồi làm thương hiệu cho nó để sau này để cho nó.
speaker_08: Trời ơi, đã chưa? Bởi vì ứng gì có chăm mẹ chào. Rất là sợ.
speaker_05: Là từ đời ông nội, xong tới đời ba, xong tới đời em thì em học rồi em đi làm điện.
speaker_05: Đi làm điện. Em nhận công trình, em làm kỹ sư bên điện.
speaker_04: Mà lúc đó gia đình có cấm cản, con đừng có làm ngành điện, làm nghề vàng đi con.
speaker_05: Có chứ. Nhưng mà sau này em mới nhận ra được là nhà em có tiệm cho nên em về để em làm.
speaker_05: Ở trước em nhìn chưa ra.
speaker_08: Thôi em nhận ra nhà em có tiền nè.
speaker_04: Giống như đi làm điện là mấy năm trời mới sắm được chỉ vàng. Trong khi nhà mình nguyên tiệm vàng không làm.
speaker_06: Ê, nhưng mà cái đó thú vị đó, cái này là hỏi thiệt, chứ không có giỡn là lý do tại sao mà tới ngần đó. Tụi có nghề, học rồi ra đi làm, rồi mới biết là nhà mình có điều kiện kìa. Cái cách giáo dục ở nhà như thế nào hay là như thế nào mà mình không nhận ra cái điều đó.
speaker_05: Lúc đó là mình đi học là mình thích cái gì mình học cái đó. Thì mình kiểu như là cho tự động mình phát triển rồi mình muốn đi đâu. Mình buông ba một thời gian vậy đó. Có nghĩa là bố mẹ cũng không cho tiền tiêu nhiều luôn hả? Hay gì sao?
speaker_05: Dạ, xin mà cũng không có cho...
speaker_08: Như lại do gia đình cũng muốn con mình tự lập, vậy thì lúc mà Duy làm điện làm bao lâu?
speaker_05: Dạ em làm cũng sáu năm.
speaker_06: Là học ra là tính ra là 21, 22 tuổi. Làm 6 năm là tới 28 tuổi mới biết nhà giàu. Dạ.
speaker_08: Rồi về lại cái nhà này được bao nhiêu năm rồi?
speaker_05: Em mới về đảm đương được chừng 2 năm.
speaker_05: Rồi giờ mình kiểu như là mình không làm nghề, nhưng mà mình kinh doanh buôn bán bên lĩnh vực này. Thì em phát triển bên cái mảng online. Thì bữa nay em lên trong chương trình là em muốn chia sẻ cái...
speaker_05: Cái ngành vàng bạc nữ trang này mà bây giờ nó phát triển hình thức online.
speaker_08: Wow! Vàng bạc mà sao bán online được nha.
speaker_06: Thì bây giờ mới thú vị đó không ai bán, có người bán người ta mới mua.
speaker_08: Nhưng mà vậy anh chia sẻ nhiều hơn đi, Duy chia sẻ nhiều hơn về cái...
speaker_05: Thì kiểu như mà qua cái khoảng thời gian mà mấy năm dịch đó. Thì người ta sẽ nằm nhà, người ta mua hàng không à mà. Thì vàng bạc nữ trang cũng y chang vậy luôn chứ. Thì họ sẽ kiếm một cái đơn vị nào uy tín.
speaker_05: Họ biết cái cửa hàng nằm ở đâu rồi. Thì họ chỉ book online thôi. Cái là có người lên live, em cũng lên live stream luôn. Em lên em live thì bao nhiêu chỉ là cân nặng, tiền công, bao nhiêu vậy đó. Thì mình sẽ có một cái đơn vị vận chuyển mà mình...
speaker_05: Mình tin tưởng họ sẽ lại họ giao, khách hàng họ sẽ mở ra họ coi, thì họ ok, đúng cái trang vậy thì họ nhận thẳng được tiền.
speaker_08: Nó xóa bỏ được định kiến về cái chuyện mà mua vàng là phải ra tiệm và phải coi đồ này nọ hả? Bây giờ ví dụ Duy muốn bán cái vòng này trên online thì Duy sẽ nói thế nào.
speaker_08: Ví dụ như giờ mình đang live nè.
speaker_05: Ok, thì mình sẽ nói với khán giả của mình là kiểu như là mọi người ở trên tay mình đang cầm là một cái vòng thương hiệu, mọi người nhìn cũng biết rồi.
speaker_05: Nếu mà trên live mình nói là sẽ sập trên mình đó, thì cái vòng này nó chừng nặng khoảng là 3 chỉ.
speaker_05: Dạ. Dành cho những người mà size tay từ 48 cho tới mà 52 thì cũng sẽ đeo được nha mọi người. Thì ở đây là nó có giấy kiểm định của bên kiểm định thứ ba.
speaker_05: Mọi người nhận hàng, mọi người xem rồi đúng trọng lượng, đúng tiền công, trọng tiền là... Thì mọi người thanh toán, mọi người có thể đeo, thử. Nếu mọi người thấy có hợp với mình.
speaker_05: Trên live thì nhiều khi là mình không biết là nó có hợp với mình hay không. Nhưng khi mình về nhà mình đeo, đeo bông tai, mình soi gương, mình thấy ok, thì mình mua còn không thì không sao. Miễn chiếc không gì hết với ai mọi người.
speaker_06: Nhưng mà có vấn đề là chắc là như phải có một thời gian để xây dựng lòng tin của khán giả. Chứ còn người ta mới coi cái live đầu tiên chắc là chưa dám mua đâu. Mà phải có một thời gian gì thì cái thời gian...
speaker_05: Đó là Duy đã làm gì? À em xây kênh. Lúc đó mình quay những clip là mình quay xưởng mà nó nói. Ờ nè ba tui đang làm nè. Ờ chú tui đang làm gì nè. Thì mình tạo lòng tin từ từ. À thằng này nó làm nghề. Ừ. Thằng này không phải là kiểu như là.
speaker_08: Là cha có cơ sở có xưởng đồ thiệt.
speaker_05: Dạ. Thì lúc đó mình lên, mình đầu tiên cũng tính live sau này. Mọi người mình lên live bán thì mọi người cũng ủng hộ. Lên live bán là mua.
speaker_08: Hay quá ha. Vậy thì có câu chuyện vui nào mà... Có trường hợp nào để chia sẻ với mọi người.
speaker_05: Chuyện vui bên ngành này sẽ nói nhiều lắm chị. Thì bên này em có cái xưởng xi mạ. Xi mạ thì anh biết rồi. Ví dụ như cái vòng hồi nãy đi. Chị Như. Vàng bạc đúng không?
speaker_05: À. Thì những khách hàng qua bên nhà em có những khách hàng mà họ đóng hụi chết của em hàng tháng. Là không phải chơi hụi với em nha.
speaker_05: Dạ, đúng rồi. Tại vì trang sức mà bạc thì mình xi mạ lên mà vàng thì nửa tháng khoảng một tháng là mình phải xi lại liền.
speaker_05: Mà cái câu chuyện của em kể đây là nó cũng eo le lắm.
speaker_05: Mà mấy anh chị cũng biết là... Con gái có chồng thì lúc nào cũng sẽ có một cái bộ vàng cưới. Ừ. Vàng cưới. Và cái bộ vàng cưới đó thì kẹt, túng thiếu lắm mới đem ra bán thôi.
speaker_05: Chứ không bao giờ đụng đến đó, thì thường thường sẽ cất trong tủ. Đó. Thì khách hàng của em là có mấy anh, mấy chị, nhiều khi có mấy bà già chồng, mấy bà già vợ luôn.
speaker_05: Cửa hàng của em cũng có cầm đồ.
speaker_05: Thì cũng có bán những cái món đó luôn. Thì cầm thì ai đem lấy mình cầm thôi mà đúng trọng lượng, đúng này kia mình cầm thôi. Tiệm... đem lời cầm! Thì những khách hàng đã cầm thành ra quen. Mình nhiều chuyện mà mình hỏi là.
speaker_05: Ủa sao vậy? Nói trời ơi con ơi hay là em ơi vậy đó. Bây giờ để mấy cái này trong tủ nó đâu sinh ra lời rồi. Lúc đầy vàng nó lên rồi. Em đi cầm. Nói cầm để làm ăn bùm túm một chút. Nó cũng vậy ha.
speaker_05: Xong cái đi tìm đi cầm, cầm xong cái mà nói nha. Em làm giùm anh một bộ y hệt cái này nè. Kiểu như là ai cưới chồng có vợ mà giàu thì bèo gì cũng 1 cái vẻ.
speaker_05: Còn mà thường thì có người là cầm em 5-7 cây vậy đó, 5-7 nhiều lắm. Nói là em phải làm bạc giống y chang như thế nha. Kiểu như cái vòng vàng như thế này là lại làm giống y hệt như vậy luôn. Ở trong là bạc xi vàng ngoài. Dạ đúng rồi.
speaker_05: Thường thường là em sẽ 1 cái 1 tháng đi. Thì... Cái trang sức này nó sẽ bị ố vàng, bạc màu, anh...
speaker_05: Chị là mình phải làm lại, nhưng mà nửa tháng lại lại rồi, tại kiểu như là... Nằm đi mà ngủ ngon á.
speaker_05: Thì đó. Thì mới lại là xi lại.
speaker_05: Nó cho nó mới, nhưng mà vui lắm nha! Lạ hả, là gửi quán cà phê nè, gửi bà bán cá chứ gửi nhà em mà chứ không dám vô thẳng vô nhà. Là tại vì vòng vòng ở khu vực của năm đó là kiểu như sợ vợ đi gặp, hay là chồng đi gặp vậy đó.
speaker_05: Nói ra gửi mà đó cái điện, à lô, đi anh để, ngoài bàn cá, mà trái, mà vô xi, mà đeo khó vàng, í, mình kín mít.
speaker_06: Vậy là anh nghĩ em nên mở thêm cái quán cà phê kế bên nhà luôn. Hả.
speaker_06: Để cho khách vô uống người ta có đường.
speaker_06: Người ta có đường người ta đi lòn. Vô phía trong người ta gửi luôn cho nó tiện em.
speaker_05: Đúng rồi, để em suy nghĩ đến cái chuyện đó.
speaker_08: Là có nghĩa là bây giờ người ta cầm cái bộ thiệt. Người ta để cái bộ kia ở nhà chơi vui rồi.
speaker_05: Dạ đúng rồi, để mà khi mà chị dâu hoặc anh chồng là mở thì thôi ồn nó thộn đâu. Ờ sao mà giống đời cô lự vậy trời.
speaker_08: Giờ tháng thì về khoe với vợ là thật ra anh có câu chuyện muốn khai thiệt với em. Rồi còn nếu như mà mình thất bại quá thì mình sẽ cố gắng mua lại cho bà vợ chúng ta bằng được. Ah...
speaker_05: Mà bà vợ không dám đi cầm bắt đầu biết đâu vậy? Đúng rồi, đúng rồi, đúng rồi. Nhưng mà cái câu chuyện hay vậy? Em có cái khúc đó nữa. Anh cầm đi cũng quen.
speaker_05: Xong tới bà chị đó mới cầm. Bởi là đem đúng cái món đồ giả của mình là một thau của các bạn. Một thau lượng. Ở đây nhân viên mình cầm nữa. Và lúc đó mình không có nhà.
speaker_05: Mà hôm đó em đi vắng đi về, nhân viên nhà mình nó chỗ chỉ với cái này đâu dạ. Để lúc nát! Em về, em thấy, quen mà.
speaker_05: Tìm kìa. Hôm mình về, mình kẹp, mình gặp, thiệt, thiệt, thiệt đấy em, thiệt đó. Trời ơi, anh chị quen, em mới hốt, em mới lên trên lầu. Em ơi gọi cho anh.
speaker_05: Anh tưởng tượng hôm đó anh sẽ nói gì với em.
speaker_08: Rồi sao? Rồi cuộc cờ sao?
speaker_05: Cuối cùng là anh không biết là anh đi xoay họ hàng, gom hết mọi nơi đâu cũng đủ. Tại vì em đâu có dám đưa 2 lần.
speaker_05: Thì mình nói, cái nhẫn này lên, mình nói, ờ cái này thiệt á chị. Rồi xong rồi mình mới đưa tiền cho chị. Về lúc đó là gia đình mới yên ổn mà chị.
speaker_08: Trời ơi, vậy là anh có nghĩa là Duy làm một pha là cứu vớt gia đình người ta.
speaker_06: Mà từ câu chuyện này của em...
speaker_06: Gút ra được một cái thông tin. Là em là chắc cái tiệm cầm đồ lớn nhất đó luôn ai cũng lại kiếm em. Chứ ông chồng cũng kiếm em mà bà vợ cũng kiếm đúng chỗ của em.
speaker_05: Mà cũng kiếm đúng ngày em thôi.
speaker_05: Tại vì bên dưới hàng có 1 cái là mua ở đâu, mình bán đó hoặc là mình cầm nó nó không có giá. Còn kiểu như là tiệm của anh mà em mua, em đi bán cho tiệm chị Như.
speaker_05: Vì như với nó, ở cái này mối này của ai, đừng...
speaker_05: Mày đúng tuổi không? Mày đi đâu, Phổ?
speaker_04: Tụi mình tụi mình tụi mình tụi mình tụi mình tụi mình. Đó, vậy em tưởng bán vàng là khỏe lắm chứ, ngồi trong mát, bán vàng tỉnh queo thôi chứ. Ai ngờ đâu bán vàng phải biết diễn nữa.
speaker_08: À, nói tới diễn nè, nói tới diễn là có cái mini game này dành cho mọi người nè. Ủa, có lắm. Chạy ra, lấy đi cưng. Diễn đây.
speaker_05: Ủa, cái mini game gì có mũ?
speaker_08: Đây, cái trò chơi này rất là dễ thương luôn. Muốn vàng, muốn bạc. Rồi, mình sẽ để cái che mắt vậy nè.
speaker_08: Với kêu vàng, làm cho mở ra.
speaker_08: Chứ kêu bạc là mở ra. Vàng.
speaker_08: Bạc. Vàng. Bạc. Bạc. Vàng. Đây là phần thưởng.
speaker_08: Nó có rất là nhiều loại nước ở trong này. Nếu như mình sai, mình sẽ uống.
speaker_06: Chơi bên đây là cứ uống cái nước này.
speaker_08: Cái gì vậy? Cái gì vậy?
speaker_08: Thôi không sao, uống được ly rồi. Cho uống miếng nước nha. Dạ, xin mời quý vị khán giả cùng nghỉ lao trong ít phút. Chúng tôi sẽ quay trở lại ngay.
speaker_05: Bên cái mảng mà kinh doanh bán nữ trang vàng bạc. Thì em có rất nhiều bạn bè anh em. Thì cái lúc mà em chuyển đổi qua cái... chỗ đó mà em đem bán livestream bán hàng. Thì mọi người nói.
speaker_05: Bạn em đang nghỉ chơi hết. Đúng đúng rồi. Cái giá nó rẻ hơn người ta nhìn. Thì bán bèo mà nói là bán phá giá.
speaker_06: Rồi thì cứ nghe hiệu lệnh lật nè. Với ai lật sai và lật chậm gì đó thì là thua nha.
speaker_04: Kìa, hay. Hay thiệt luôn, thần kinh, anh Tiến kìa. Khủng đi em. Ok.
speaker_04: Đi nó ngọt á, tới anh đi.
speaker_06: Cái này đáng đời cho anh nè. Cà phê. Ủa vậy hả? Còn 2 ly kìa Như. Tiến, Như thôi, đứa 1 ly đi. Khoa khỏi chơi nữa, đứa 1 ly nè cho em.
speaker_08: Không, không, em dẹp luôn cho em, dẹp luôn, thôi con, đùa đủ rồi. Hồi nãy mình chơi để vui thôi, về cái độ nhạy, thật ra...
speaker_08: Vàng bạc đó chỉ là một cái thói quen, cái phản xạ thôi. Tại vì anh Tiến chơi hơi bị giỏi, ha?
speaker_06: Một xuyến là mua vàng, mua bạc hay sao? Không, em ở ở nhà anh chọn kèm, em đổ qua chiếc này, vô qua chiếc kia, vô đây em. Đổ lộn là hư hết nha tao, anh đâu được phép lộn với em.
speaker_08: Phải đúng, phải đúng nha chứ.
speaker_08: Trời ơi. Bây giờ vấn đề mà anh mang tới chương trình đó là gì à.
speaker_05: Vấn đề em mang đến với mọi người để mà trao đổi là. Kiểu như là. Bên cái mảng mà kinh doanh và bán nữ trang vàng bạc đó. Thì em có rất là nhiều bạn bè anh em. Thì cái lúc mà em chuyển đổi qua cái.
speaker_05: Chỗ đó mà em để bán livestream bán hàng. Hoặc là mọi người nói.
speaker_05: Bạn bè mình đang nghỉ chơi hết. Đúng, đúng rồi.
speaker_05: Đúng, anh nói đúng rồi, hợp lý luôn anh. Tại vì một cái mặt bằng mà chị Như với hai anh nghĩ là phải tốn mấy chục triệu một tháng. Rồi nhân viên, điện nước, các kiểu luôn.
speaker_05: Thì thí dụ một món em nhập về 10.000, em phải bán nó cao hơn chứ.
speaker_05: Để mà trang trải được cái món đồ là kiểu như nó phải bù vô những cái đó đúng không? Còn khi mà mình đưa những sản phẩm này lên, thương mại điện tử, bán online. Tạm biệt!
speaker_05: Mình sẽ, cái giá nó rẻ hơn người ta nhiều. Thì bạn bè nói là bán phá giá.
speaker_08: Mà bạn bè có kiểu can thiệp hay là có những lời lẽ nào để tác động tới mình không?
speaker_05: Thì nói chung là cuộc sống thì bạn bè tự nhiên mạnh ai làm nấy sống vậy đó nhưng mà thực sự là gắn bó trong cái nghề. Nhưng mà mình nói với bạn mình thì bây giờ chuyển đổi bên đó làm cái đó đi, mình thấy nó hiệu quả.
speaker_05: Mọi người nói, tao không phải như mày, tao làm không được. Nói như mà cái nghề đó. Nói như là mình nói với mọi người là có thể thuê người làm. Mình có tiền, bỏ thuê người ta làm, mình chuyển đổi.
speaker_05: Chứ kiểu như 3 anh chị cũng hiểu là 1 ngày mà 1 cửa hàng. Thí dụ vô mà chừng 100 khách đi thì tiếp cũng mệt mỏi lắm á.
speaker_05: Còn cái vụ, một cái màn hình online. Có thể tiếp cận vài ngàn người và chục ngàn người. Trên một phiên live có 2-3 tiếng đồng hồ thôi.
speaker_05: Thì cái sản phẩm của mình, thí dụ một sản phẩm mình. Mình nhập về mười mẫu đi. Mình bày mười, người vô cửa hàng mình mua trời, ngồi chờ, ngồi mỏi luôn. Mà mình ship cho 10 mẫu đó lên chừng không tới 5-10 phút đâu. Mà bán 1 cái giá hợp lý, nó rẻ hơn ở ngoài cửa hàng thôi.
speaker_05: Thực sự có nhiều người, nhiều người bạn bè mình thôi, không hiểu quá chị.
speaker_06: Mà anh nghĩ vậy nè. Cái việc em làm á. Thật ra là...
speaker_06: Nó không có giành quá nhiều thị phần của những tiệm vàng của bạn bè. Nên em ngược lại mà em mở được thị trường mới. Nghĩa là khi em có một cái tiệm vàng thí dụ ở quận 5 đi. Thì thật ra những người họ có nhu cầu.
speaker_06: Và họ biết tới tiệm em họ mới ra tiệm.
speaker_06: Nhưng khi em làm livestream kể cả những người không có nhu cầu. Giờ đang lướt Facebook, đang lướt TikTok. Thấy là thèm là mua luôn.
speaker_08: Em là chuyên gia nè. Em hãy coi mấy bạn mà live về đá, về mấy cái viên đá, hồng ngọc đồ này.
speaker_08: Coi như là em hãy xem lắm, tại vì chỉ có livestream mọi người mới cho coi nhiều đồ mà mình không có thời gian ra. Và em cũng có mua online, nhưng mà em mua...
speaker_08: Online ở cái tiệm quen, ví dụ như những ngày mà vía Thần Tài hay gì đó này nọ đó. Thì mình được cái mình đang quen, mình không có cái mình kêu ship cho chị đi hay chị kêu vậy. Thì thành ra người ta cũng sẽ ship.
speaker_08: Mà quan trọng là cái mà Duy làm được đó là cái quen thân.
speaker_08: Trang sức là uy tín là quan trọng.
speaker_08: Trường hợp của Duy là khi mình làm cái nghề này mình có sự vui sướng. Vì mình mở rộng thị trường của mình.
speaker_08: Tuy nhiên mình lại đóng lại những mối quan hệ bạn bè. Thật sự trong lòng của Duy á, Duy có muốn thuyết phục bạn bè là vào cái thị trường mới của Duy đang mở ra hay không?
speaker_08: Hay là Duy chỉ muốn là mối quan hệ mình đừng có nghỉ chơi với nhau? À.
speaker_05: Em suy nghĩ rất là nhiều. Em mở cái tiếp thị liên kết là affiliate đó.
speaker_05: Bên ngành nữ trang vàng bạc luôn, em mở 40%.
speaker_05: Thì em nói bạn bè em nó thích dùng gì, ai ngại, hổ dốn hay là tất cả mọi người luôn. Tất cả mọi người luôn, mình nữ trang vàng bạc. Thì cứ liên hệ em để em đưa sản phẩm cho người ta về. Cứ lấy đi, thí dụ không được thì trả lại không mất tiền.
speaker_08: Nói là nếu mà bán tại cửa hàng với bán online thì tỷ lệ phần trăm lợi nhuận mình thu được, nó là bao nhiêu?
speaker_08: Cái gì dữ vậy đó hả?
speaker_06: Vì sao, đối với kinh doanh mà có một cơ sở có một mặt bằng á, nó sẽ có một cái trần doanh thu. Có nghĩa là với cái cơ sở đó mặt bằng nhiêu đó, diện tích nhiêu đó và bấy nhiêu đó nhân công, tối đa thì doanh thu cũng có.
speaker_06: Nhưng đối với thương mại điện tử là nó không có giới hạn luôn, có nghĩa là mình càng phát triển càng lớn nó sẽ càng lớn và nó không có bị cái giới hạn của cái phần đó. Cho nên là hầu hết mọi người đổ về thương mại điện tử là vậy.
speaker_04: Ủa mà vậy thì hay quá, mà sao mà các bạn không làm theo ta?
speaker_08: Tại vì người ta ngại đổi mới.
speaker_06: Mà cái xu hướng bây giờ anh thấy nha. Đa phần khách hàng là họ thích trong việc kể cả mình mua đồ ăn mình cũng thích coi người ta nấu làm sao.
speaker_06: Mình mua cái gì mình cũng thích cái quá trình nó tạo ra như thế nào. Thế nên anh nghĩ Duy làm được cái đó có nghĩa là. Thật ra hồi đó giờ mua vàng đi. Thứ vàng mua rồi đâu có biết sao nhưng mà ông này mà ông làm những clip hướng dẫn là. Vàng tạo ra thế này nè nọ nọ kia.
speaker_06: Mình chơi vàng mà mình có kiến thức mình nói chuyện mình khoe ba má mình khoe với người thân. Thì tự nhiên mình thấy thích, thì từ cái kênh bắt đầu là với mục đích chia sẻ kiến thức. Thì người ta coi, người ta thấy ông này cũng có kiến thức. Xong bắt đầu có lòng tin, tin rồi bắt đầu mới chuyển đổi thành cái mua hàng.
speaker_05: Thì nó phải tốn 1 cái quá trình. Kiểu như là 1 chiếc nhẫn xoàn. Người ta có thể bỏ vài chục người ta mua. Ta thấy đẹp là được. Nhưng người ta không biết là cái công ngồi chế tác. Cực khổ lắm. Nhiều khi 1 clip em làm mấy phút. Làm 3 ngày đó.
speaker_05: Nó lâu dữ lắm chị. Từ 1 cục vàng vậy mà làm thành 1 chiếc nhẫn luôn. Còn kỳ cọc không được làm như vậy.
speaker_08: Hông, cái đó hay thiệt. Duy đã mang đến một cái gọi là lại thêm một cái cảm hứng về kinh doanh. Nhưng mà một cái điều quan trọng hơn nữa như tinh chắc là...
speaker_08: Cái gì cũng vậy. Tất cả mọi thứ mà mình làm hay là mình mua hay mình chọn, nó đều phải có sự chọn lọc.
speaker_08: Sự uy tín, sự chất lượng, sự kỹ càng. Và điều hơn hết là sự hiểu biết.
speaker_08: Giống như Duy bán Duy có sự hiểu biết về sản phẩm của Duy. Cái người mua cũng có sự hiểu biết về sản phẩm. Để mà có thể tìm được, chọn lọc được. Cũng như mình kinh doanh mình cũng chọn đối tác là một người rất là uy tín.
speaker_08: Nên mà sau cái ngày hôm nay. Hôm nay là Như Mộng là tất cả Đăng. Cả em, còn anh Tiến thì em không nói rồi. Anh Tiến cũng là một người thành công. Về cái thị trường.
speaker_08: Buôn bán là một người tiên phong về rất nhiều lĩnh vực. Anh thấy anh cũng là một người thành công, rất là đáng học hỏi. Là Như Mộng. Chứ tính cả mọi người, ai cũng sẽ tìm được cho mình một cái...
speaker_05: Tại vì em nói không lại ba anh chị.
speaker_05: Nhưng mà nhìn mặt của anh gian xảo hơn em thì sao?
speaker_08: Đó là cái gene buôn bán nữa.
speaker_08: Vậy thì nhân đây có lời nhắn gì với những người bạn cùng ngành của mình không?
speaker_05: Thì cũng nhân đây, thông qua cái chương trình này thì cũng gửi lại các đồng nghiệp. Với lại các bạn bè làm chung trong cái ngành mình á. Thì cái ngành kim hoàn hồi xưa tới bây giờ là nó có cái nghề rất là đặc biệt.
speaker_05: Nhưng mà ít cái người biết tới á. Thì thường thường là ai làm gì làm mà quay phim hay là chụp hình cũng khó lắm. Tại vì người ta không muốn xuất hiện nữa. Tại vì người ta đứng sau kính vè hà là giáo đồ thôi.
speaker_05: Chứ bây giờ không cho ai biết là sản phẩm này ai làm hết. Thì nói chung. Mình làm được những sản phẩm tốt. Bên ngành kim hoàn ở Việt Nam. Mình rất là phát triển khủng khiếp. Những món đồ quốc tế người ta làm được thì Việt Nam mình làm được hết.
speaker_05: Thì... Mình cũng có thể chuyển đổi qua mạng online. Với lại là mình... Mở rộng.
speaker_05: Mở rộng thêm. Nhưng lại là phải tự tin lên. Cảm ảnh đó. À, tui nè. Sướng nè. Tui làm cái này nè. Đúng rồi.
speaker_05: Kiểu như là đừng có khép nép hay về gì cả. Nói chung là thật sự mọi thứ là mình làm được. Mình tự tin là mình đang làm những cái điều đó.
speaker_08: Dạ, đúng rồi. Đúng rồi, thời đại công nghệ số mà ai cũng muốn là cái gì nó cũng sẽ nhanh chóng. Đó là cũng là một cách để mình phát triển thị trường của mình hơn.
speaker_06: Mà anh thấy Duy làm vậy là nó đúng và nó kịp thời, bởi vì biết sao không. Hiện giờ anh thấy nha, cái thế giới này nó phẳng tới mức là nhiều khi bạn bè của anh mua 1 cái món nữ trang nước ngoài có thể order cho trang thương mại điện tử nước ngoài nó gửi về luôn.
speaker_06: Thì nếu như mà tại Việt Nam của mình không làm được cái chuyện đó. Thì mình mất hết toàn bộ giá trị về nước ngoài. Mình mất hết toàn bộ cái nguồn thu đó cho nước ngoài. Mà chính anh bây giờ. Thì kể cả ở cái ngành mỹ phẩm của Việt Nam đi.
speaker_06: Thì chính anh cũng phải đang tự mình làm cái chuyện đó để mà... Thay vì người Việt Nam xài mỹ phẩm nước ngoài thì anh phải cho người Việt Nam xài mỹ phẩm của Việt Nam. Thì dân mình mới giàu được. Thì anh cảm thấy rất là...
speaker_06: Thích thú và cảm ơn Vy, vì Vy là người có thể nói là ở khu vực của Gi đi. Vì Gi là người tiên phong làm điều này và cái lời này cũng sẽ là một lời cảnh tỉnh. Bởi vì đã tới lúc thời đại phẳng rồi, mình phải làm cho nó phẳng thật sự.
speaker_06: Chứ không thể nào để mình đổ tiền ra nước ngoài như vậy được.
speaker_08: Dạ, và những cái lo lắng của Duy. Cũng giống như là những cái trăn trở của Duy ngày hôm nay. Không biết là cái buổi nói chuyện ngày hôm nay có làm cho Duy cảm thấy thoải mái hơn không?
speaker_05: Vì em thoải mái. Hả, cái buổi nói chuyện ngày hôm nay thì em cảm ơn anh Tiến. Đăng và chị Như. Chứ chung là đã... Để cho em kể hết cái nỗi lòng của em ra cho mọi người cùng nghe ở trên đây.
speaker_05: Thì cũng mong là sau này kiểu như là tất cả mọi người sẽ có 1 hướng đi mới từ ngày hôm nay em chia sẻ vậy nè. Thì cái thị trường này có nhiều khi người ta làm được nên ta không nói đâu.
speaker_05: Còn nhiều khi làm được mình nói cho mọi người biết để mọi người cùng tìm hiểu với cùng làm. Thì qua chương trình này mình cũng mong là cái ngành kim hoàn, nữ trang đá quý bên ở Việt Nam mình.
speaker_05: Và cũng là anh em trong ngành nghề của mình sẽ...
speaker_05: Ngày càng phát triển hơn. Dạ. Cái lại là cái ngành online mà bên nữ trang vàng bạc. Nó sẽ không thua kém gì bên thị trường nước ngoài.
speaker_08: Wow! Cảm ơn Vy. Cảm ơn Vy ngày hôm nay. Cảm ơn Anh Tiến. Cảm ơn Đằng. Vì chúng ta đã cùng nhau trong một cái buổi nói chuyện với những chủ đề khác nhau.
speaker_08: Nhưng mang rất là nhiều, nhiều cảm hứng. Và xin một lần nữa, cảm ơn quý vị khán giả. Đã theo dõi chương trình Chuyện Họ Chuyện Mình. Quý vị thân mến, chương trình do Đài Truyền hình Thành phố Hồ Chí Minh phối hợp cùng với Madison Media Group thực hiện.
speaker_08: Được phát sóng vào lúc 20h thứ Bảy hằng tuần trên kênh HTV bảy. Còn bây giờ, xin chào và hẹn gặp lại!
```

#### 🤡 [TỪ MÔ HÌNH BASE CHƯA HỌC]:
*(Nó thường hay chép lại văn bản gốc hoặc chém gió)*
```text
# 1. TỔNG QUAN CUỘC HỌP
Chương trình "Chuyện Họ Chuyện Mình" do Đài Truyền hình Thành phố Hồ Chí Minh phối hợp cùng Madison Media Group tổ chức. Chương trình có sự tham gia của khách mời Phương Oanh, Dược sĩ Tiến, và Hữu Đằng. Nội dung chương trình xoay quanh các vấn đề về giảm cân, tự tin, và kinh doanh online trong ngành kim hoàn nữ trang.

# 2. HÀNH ĐỘNG TRIỂN KHAI (ACTION ITEMS)
- Phương Oanh chia sẻ câu chuyện về việc giảm cân và tự tin với ngoại hình.
- Dược sĩ Tiến và Hữu Đằng đưa ra lời khuyên về việc giảm cân và tự tin.
- Phương Oanh nhận được lời khuyên và động lực từ khách mời và khán giả.
- Dược sĩ Tiến chia sẻ kinh nghiệm kinh doanh online trong ngành kim hoàn nữ trang.
- Phương Oanh và Dược sĩ Tiến thảo luận về việc chuyển đổi sang kinh doanh online và tác động của nó đến thị trường.

# 3. CHI TIẾT THEO NGƯỜI NÓI (SPEAKER INSIGHTS)
- **Khả Như**: Tạo không khí vui vẻ, tạo cơ hội cho khách mời chia sẻ câu chuyện và đưa ra lời khuyên.
- **Phương Oanh**: Chia sẻ về quá trình giảm cân, tự tin với ngoại hình, và động lực để vượt qua những khó khăn.
- **Dược sĩ Tiến**: Chia sẻ kinh nghiệm kinh doanh online, tư vấn về việc giảm cân và tự tin.
- **Hữu Đằng**: Đề xuất các giải pháp và động viên khách mời.

# 4. KỊCH BẢN CHUẨN HOÁ (CLEANED TRANSCRIPT)

[13.8s - 25.0s] Khả Như: Xin chào quý vị khán giả, quý vị thân mến, chương trình Chuyện Họ Chuyện Mình do Đài Truyền hình Thành phố Hồ Chí Minh phối hợp cùng Madison Media Group thực hiện. 

[25.0s - 36.9s] Khả Như: Như thường lệ, Như sẽ đồng hành cùng một câu chuyện của mình và khách mời. 

[37.1s - 43.2s] Khả Như: Mục đích cuối cùng của chúng tôi là mang lại những giây phút thoải mái cho quý vị. 

[68.4s - 70.6s] Khả Như: Đằng, chị sợ quá, chị sợ quá.

[72.5s - 76.7s] Khả Như: Đằng, chị sợ quá, chị sợ quá.

[78.0s - 82.3s] Khả Như: Tại vì chị sợ quá, chị sợ quá.

[82.3s - 83.4s] Khả Như: Trời ơi, em có làm gì đâu.

[85.8s - 89.8s] Khả Như: Dạ, là cái này là cái số của em hay sao vậy Đằng? Số bảy... ờ?

[92.7s - 94.4s] Hữu Đằng: Không có, tại số 7 là số hên của em.

[94.5s - 97.3s] Khả Như: Ai vậy hả? Chứ tưởng là em là đại diện cho số bảy.

[100.2s - 104.8s] Hữu Đằng: Dạ, Hữu Đằng xin gửi chào quý vị khán giả đang xem chương trình Chuyện Họ Chuyện Mình.

[107.7s - 116.5s] Hữu Đằng: Dạ, hôm nay có thêm một người bạn sẽ cùng lan tỏa năng lượng tích cực. Dạ, xin mời anh Dược sĩ Tiến.

[128.7s - 133.4s] Đằng: Bữa nay Hữu Đằng mời anh tới đây để cho anh được trị bệnh chính chuyên.

[136.5s - 137.7s] Khả Như: Trời ơi, cái bệnh gì mà nghe ghê nữa.

[144.5s - 148.6s] Đằng: Bình thường không bao giờ tham gia một show vui, giờ vô đây không biết làm sao để vui luôn.

[154.0s - 156.9s] Khả Như: Còn khi lên sóng là tất cả mọi người kiểu như đi họp.

[166.4s - 169.7s] Khả Như: Có khi là lên sóng thì chỉ còn có em với anh thôi, chứ không có Hữu Đằng luôn.

[172.1s - 174.3s] Đằng: Dạ đúng rồi, biên tập dựng vậy đi.

[181.7s - 189.0s] Khả Như: Nhưng thực sự khi người ta mời cả ba chúng ta cùng ngồi đây thì chắc chắn khách mời ngày hôm nay phải có vấn đề.

[192.0s - 193.5s] Hữu Đằng: Love, chính chuyện còn hơn anh.

[193.7s - 196.0s] Khả Như: Không phải chính chuyên đâu, em nghĩ là mình nói không nổi.

[196.3s - 197.8s] Hữu Đằng: Là một vấn đề lớn hay sao chị?

[200.4s - 202.3s] Khả Như: Mà cả mọi người tới đây đều có vấn đề lớn.

[205.2s - 207.0s] Hữu Đằng: Em luôn muốn gặp những nhân vật đó quá.

[207.2s - 209.0s] Khả Như: Đúng rồi, nhân vật trong chiếc hộp này.

[212.0s - 213.5s] Khả Như: Em lại chiếc hộp đi, em lại chiếc hộp đi.

[214.6s - 216.3s] Hữu Đằng: Ê, có khi nào Như lại hù em không?

[216.2s - 217.6s] Khả Như: Rồi chị đếm 1, 2, 3 nha.

[227.3s - 231.8s] Khả Như: Quý vị thân mến, trước khi đến với chiếc hộp này thì chúng ta cùng nghỉ giải lao trong ít phút. Chúng tôi sẽ quay trở lại ngay.

[247.9s - 255.0s] Khả Như: Quý
```

#### 🏆 [TỪ MÔ HÌNH ĐÃ GẮN NÃO PEFT]:
*(Nó sẽ đóng khuôn cứng ngắc các Header #)* 
```text
# 1. TỔNG QUAN CUỘC HỌP

Cuộc họp/buổi ghi âm/đoạn hội thoại: "Chuyện Họ Chuyện Mình" là một chương trình phát sóng trên Đài Truyền hình Thành phố Hồ Chí Minh. Nội dung chính bao gồm các thảo luận, chia sẻ từ các khách mời và các thảo luận từ các phát ngôn viên. Nội dung chính bao gồm các thảo luận từ các phát ngôn viên, các thảo luận từ các khách mời, và các thảo luận từ các phát ngôn viên.

# 2. HÀNH ĐỘNG TRIỂN KHAI (ACTION ITEMS)

Không có hành động triển khai cụ thể được đề cập trong nội dung thảo luận.

# 3. CHI TIẾT THEO NGƯỜI NÓI (SPEAKER INSIGHTS)

## Speaker 08 (Khả Như):
- Khả Như, xin chào quý vị khán giả đã đến với chương trình Chuyện Họ Chuyện Mình. Quý vị thân mến, chương trình do Đài Truyền hình Thành phố Hồ Chí Minh phối hợp cùng với Madison Media Group thực hiện. Được phát sóng vào lúc 20 giờ thứ Bảy hằng tuần trên kênh HTV7.
- Như thường lệ, Như sẽ đồng hành cùng với một câu chuyện của mình. Của mình và những khách mời đến đây, họ sẽ có những vấn đề mà mang đến cho Như cũng như là mong là quý vị khán giả có thêm những câu chuyện cho đời sống của mình thêm thú vị.
- Mục đích cuối cùng của chúng tôi đó là muốn mang cho quý vị những giây phút thật sự thoải mái. Xin trân trọng kính mời.
- Đằng đó chị sợ quá Đằng. chị sợ quá
- Đằng, tại vì chị có nói với chương trình là nếu như mà có Hữu Đằng thì bữa đó cho chị bệnh.
- Sao vậy? Tại chị sợ lắm. Tại vì đi diễn với em về lần nào chị cũng bầm mình hết á. Chị bị sợ lắm.
- Trời ơi. Em có làm gì đâu.
- Dạ. Là cái này là cái số của em hay sao vậy Đằng? Số bảy... Ờ?
- Ai vậy hả? Chứ tưởng là em là đại diện cho số bảy.
- Trời ơi. Em có làm gì đâu.
- Dạ. Là cái này là cái số của em hay sao vậy Đằng? Số bảy... Ờ?
- Trời ơi. Em có làm gì đâu.
- Dạ. Là cái này là cái số của em hay sao vậy Đằng? Số bảy... Ờ?
- Trời ơi. Em có làm gì đâu.
- Dạ. Là cái này là cái số của em hay sao vậy Đằng? Số bảy... Ờ?
- Trời ơi. Em có làm gì đâu.
- Dạ. Là cái này là cái số của em hay sao vậy Đằng? Số bảy... Ờ?
- Trời ơi. Em có làm gì đâu.
- Dạ. Là cái này là cái số của em hay sao vậy Đằng? Số bảy... Ờ?
- Trời ơi. Em có làm gì đâu.
- Dạ. Là cái này là cái số của em hay sao vậy Đằng? Số bảy... Ờ?
- Trời ơi. Em có làm gì đâu.
- Dạ. Là cái này là cái số của em hay sao vậy Đằng? Số bảy... Ờ?
- Trời ơi. Em có làm gì đâu.
- Dạ. Là cái này là cái số của em hay sao vậy Đằng? Số bảy... Ờ?
- Trời ơi. Em có làm gì đâu.
- Dạ. Là cái này là cái số của em hay sao vậy Đằng? Số bảy... Ờ?
- Trời ơi. Em có làm gì đâu.
- Dạ. Là cái này là cái số của em hay sao vậy Đằng? Số bảy... Ờ?
- Trời ơi. Em có làm gì đâu.
- Dạ. Là cái này là cái số của em hay sao vậy Đằng? Số bảy... Ờ?
- Trời ơi. Em có làm gì đâu.
- Dạ. Là cái này là cái số của em hay sao vậy Đằng? Số bảy... Ờ?
- Trời ơi. Em có làm gì đâu.
- Dạ. Là cái này là cái số của em hay sao vậy Đằng? Số bảy... Ờ?
- Trời ơi. Em có làm gì đâu.
- Dạ. Là cái này là cái số của em hay sao vậy Đằng? Số bảy... Ờ?
- Trời ơi. Em có làm gì đâu.
- Dạ. Là cái này là cái số của em hay sao vậy Đằng? Số bảy... Ờ?
- Trời ơi. Em có làm gì đâu.
- Dạ. Là cái này là cái số của em hay sao vậy Đằng? Số bảy... Ờ?
- Trời ơi. Em có làm gì đâu.
- Dạ. Là cái này là cái số của em hay sao vậy Đằng? Số bảy... Ờ?
- Trời ơi. Em có làm gì đâu.
- Dạ. Là cái này là cái số của em hay sao vậy Đằng? Số bảy... Ờ?
- Trời ơi. Em có làm gì đâu.
- Dạ. Là cái này là cái số của em hay sao vậy Đằng? Số bảy... Ờ?
- Trời ơi. Em có làm gì đâu.
- Dạ. Là cái này là cái số của em hay sao vậy Đằng? Số bảy... Ờ?
- Trời ơi. Em có làm gì đâu.
- Dạ. Là cái này là cái số của em hay sao vậy Đằng? Số bảy... Ờ?
- Trời ơi. Em có làm gì đâu.
- Dạ. Là cái này là cái số của em hay sao vậy Đằng? Số bảy... Ờ?
- Trời ơi. Em có làm gì đâu.
- Dạ. Là cái này là cái số của em hay sao vậy Đằng? Số bảy... Ờ?
- Trời ơi. Em có làm gì đâu.
- Dạ. Là cái này là cái số của em hay sao vậy Đằng? Số bảy... Ờ?
- Trời ơi. Em có làm gì đâu.
- Dạ. Là cái này là cái số của em hay sao vậy Đằng? Số bảy... Ờ?
- Trời ơi. Em có làm gì đâu.
- Dạ. Là cái này là cái số của em hay sao vậy Đằng? Số bảy... Ờ?
- Trời ơi. Em có làm gì đâu.
- Dạ. Là cái này là cái số của
```

---
### SAMPLE DIARIZEN 3
#### 🎯 [ĐÁP ÁN GỐC TỪ CON NGƯỜI LÀM]:
```text
# 1. TỔNG QUAN CUỘC HỌP

# 2. HÀNH ĐỘNG TRIỂN KHAI (ACTION ITEMS)
Không có nhiệm vụ/deadline nào được giao trong cuộc trò chuyện này.
Tại chương trình "Chuyện Họ Chuyện Mình", Host Khả Như chào đón hai khách mời quen thuộc là diễn viên Hữu Đằng và Dược sĩ Tiến. Cuộc rôm rả xoay quanh "chứng bệnh chính chuyên" của Dược sĩ Tiến và sự thừa nhận hài hước của anh về việc bắt đầu "hư hỏng" sau những lần tiếp xúc với Host Khả Như.

# 3. CHI TIẾT THEO NGƯỜI NÓI (SPEAKER INSIGHTS)
- Speaker_08 (Host Khả Như): Giới thiệu chương trình và sự xuất hiện của hai khách mời. Ngẩn ngơ và đùa nghịch trước thông báo thay đổi tính cách đột ngột của Dược sĩ Tiến. 
- Speaker_11 (Dược sĩ Tiến): Lên án đùa rằng mình bị tiêm nhiễm "thói hư tật xấu" từ Khả Như và thông báo trịnh trọng về quyết định sẽ từ bỏ sự "chính chuyên" để trở nên hư hỏng hơn trong chương trình.
- Speaker_12 (Hữu Đằng): Gửi lời chào nhã nhặn tới khán giả và bắt nhịp tung hứng cùng các màn trêu đùa của đồng nghiệp.

# 4. KỊCH BẢN CHUẨN HOÁ (CLEANED TRANSCRIPT)
Speaker_08 (Khả Như): Khả Như xin chào quý vị khán giả đã đến với chương trình "Chuyện Họ, Chuyện Mình". Chương trình do Đài Truyền hình TP.HCM phối hợp cùng Madison Media Group thực hiện, được xuất sóng vào 20h thứ 7 hàng tuần trên kênh HTV7. Hôm nay, người bạn đồng hành của Như chính là diễn viên Hữu Đằng và Dược sĩ Tiến. Xin mời!
Speaker_12 (Hữu Đằng): Sao? Anh có sự lựa chọn vậy không?
Speaker_11 (Dược sĩ Tiến): Xin chào tất cả quý vị khán giả. Ngày hôm nay Tiến tới đây cùng với Hữu Đằng. Đằng ơi, chào một tiếng cho vui chuyện hôm nay đi!
Speaker_12 (Hữu Đằng): Dạ, Hữu Đằng xin gửi lời chào thân thương tới quý khán giả đang xem chương trình ạ.
Speaker_11 (Dược sĩ Tiến): Dạ rồi, anh vô câu chuyện của anh, anh kể đi anh. Hôm nay anh tới đây mang theo một câu chuyện... Lần trước anh tới đây, anh cũng có một câu chuyện, đó là việc anh rất chính chuyên. Nhưng mà sau một thời gian tiếp xúc với Khả Như, anh cảm thấy mình đã bắt đầu hư hỏng rồi, đúng không? Chuyện gì thói hư tật xấu là Như chỉ cho hết!
Speaker_12 (Hữu Đằng): Dạ xong hết câu chuyện luôn rồi đó.
Speaker_11 (Dược sĩ Tiến): Và để chứng minh sự chính chuyên hồi trước, anh chỉ muốn thông báo cho em biết một điều. Đó là từ bữa nay anh sẽ bắt đầu hư hỏng thật sự.
Speaker_08 (Khả Như): Khoan dạ! Đó là bảng tin thông báo chứ có phải câu chuyện trọng tâm của bữa nay đâu anh! Anh xác định lại giùm em nha anh Tiến! Mời anh vô chủ đề chính.
```

#### 🤡 [TỪ MÔ HÌNH BASE CHƯA HỌC]:
*(Nó thường hay chép lại văn bản gốc hoặc chém gió)*
```text
# 1. TỔNG QUAN CUỘC HỌP
Chương trình "Chuyện Họ, Chuyện Mình" là một talkshow truyền hình được phát sóng hàng tuần trên kênh HTV7 của Đài Truyền hình Thành phố Hồ Chí Minh và Madison Media Group. Chương trình tập trung vào các câu chuyện cá nhân, trải nghiệm và lời khuyên từ các khách mời về cuộc sống, công việc và mối quan hệ. Trong cuộc họp này, các khách mời đã thảo luận về các vấn đề như sự nhiệt tình thái quá trong công việc, cân nhắc giữa thời gian cá nhân và công việc, cũng như cách quản lý thời gian và cân nhắc nhu cầu của người khác.

# 2. HÀNH ĐỘNG TRIỂN KHAI (ACTION ITEMS)
- Kiều cần xác định lại mục tiêu cuộc sống và công việc, tập trung vào việc từ chối những công việc không cần thiết để dành thời gian cho gia đình.
- Kiều cần đặt câu hỏi cho bản thân trước khi giúp đỡ người khác, đảm bảo rằng việc đó không ảnh hưởng đến bản thân và gia đình.
- Kiều cần chắt lọc những người cần giúp đỡ và ưu tiên thời gian cho bản thân và gia đình.
- Cường cần học cách từ chối và đặt giới hạn thời gian khi giúp đỡ vợ, đồng thời giữ cho niềm vui trong gia đình không bị mệt mỏi.
- Cường cần hỗ trợ vợ trong việc cân nhắc giữa công việc và thời gian dành cho gia đình, đảm bảo rằng gia đình luôn là ưu tiên hàng đầu.

# 3. CHI TIẾT THEO NGƯỜI NÓI (SPEAKER INSIGHTS)
- **Khả Như**: Đề xuất các khách mời cần xác định mục tiêu và đặt giới hạn, đồng thời nhấn mạnh tầm quan trọng của việc cân nhắc nhu cầu của người khác.
- **Hữu Đằng**: Đề xuất các khách mời cần học cách từ chối và đặt giới hạn thời gian, đồng thời hỗ trợ người khác trong việc cân nhắc giữa công việc và gia đình.
- **Dược sĩ Tiến**: Đề xuất các khách mời cần xác định mục tiêu cuộc sống và công việc, tập trung vào việc từ chối những công việc không cần thiết để dành thời gian cho gia đình.
- **Kiều**: Đề cập đến sự nhiệt tình thái quá trong công việc và cần cân nhắc giữa việc giúp đỡ người khác và thời gian dành cho bản thân và gia đình.
- **Cường**: Đề cập đến sự ủng hộ của chồng và cần học cách từ chối và đặt giới hạn thời gian khi giúp đỡ vợ, đồng thời giữ cho niềm vui trong gia đình không bị mệt mỏi.

# 4. KỊCH BẢN CHUẨN HOÁ (CLEANED TRANSCRIPT)

[10.2s - 13.3s] Khả Như: Chào quý vị khán giả, chương trình Chuyện Họ, Chuyện Mình do Đài Truyền hình Thành phố Hồ Chí Minh phối hợp cùng Madison Media Group thực hiện, được phát sóng vào lúc 20h thứ 7 hàng tuần trên kênh HTV7. Hôm nay, người bạn đồng hành của Như là diễn viên Hữu Đằng và Dược sĩ Tiến. Xin mời!

[14.8s - 25.0s] Khả Như: Quý vị thân mến, hôm nay Như xin giới thiệu đến quý vị diễn viên Hữu Đằng và Dược sĩ Tiến.

[25.0s - 28.3s] Khả Như: Diễn viên Hữu Đằng và Dược sĩ Tiến, xin mời!

[35.0s - 36.4s] Hữu Đằng: Sao? Anh có sự chọn lựa vậy không?

[40.4s - 45.7s] Dược sĩ Tiến: Xin chào quý vị khán giả, hôm nay Tiến tới đây cùng với Hữu Đằng. Đằng chào quý vị!

[49.8s - 53.5s] Hữu Đằng: Dạ, Tiến ơi, xin chào quý khán giả của Chuyện Họ, Chuyện Mình. Hôm nay Tiến tới đây cùng với Hữu Đằng.

[55.3s - 60.7s] Dược sĩ Tiến: Dạ, anh vô câu chuyện của anh. Anh kể đi anh ạ. Anh hôm nay anh tới đây anh mang theo một câu chuyện.

[61.8s - 63.9s] Dược sĩ Tiến: Lần trước anh tới đây, anh cũng có một câu chuyện.

[66.3s - 68.5s] Dược sĩ Tiến: Đó là anh rất là chính chuyên.

[69.3s - 72.0s] Dược sĩ Tiến: Và sau một lần tiếp xúc với Khả Như.

[72.9s - 74.8s] Dược sĩ Tiến: Thì anh cảm thấy mình đã bắt đầu hư hỏng đúng không?

[76.9s - 79.0s] Dược sĩ Tiến: Trời ơi, có nhiêu là thói hư tật xấu là Như chỉ hết.

[82.8s - 84.0s] Hữu Đằng: Dạ, dạ, xong hết câu chuyện rồi.

[88.1s - 92.2s] Dược sĩ Tiến: Và nếu anh chứng minh sự chính chuyên ấy, anh chỉ muốn thông báo cho em biết là bữa nay anh bắt đầu hư hỏng.

[92.7s - 96.8s] Khả Như: Dạ, đó là thông tin chứ phải câu chuyện đâu, anh xác định lại giùm em nha anh Tiến. Em...

[96.9s - 99.0s] Dược sĩ Tiến: Mình có mở bài đàng hoàng là từ hồi tập trước.

[99.4s - 110.8s] Dược sĩ Tiến: Cái câu chuyện tập trước. Cái tập trước anh gặp em. Gặp em xong lúc đó chính chuyên. Chính chuyên xong là mới bắt đầu bớt chính chuyên. Tới ngày hôm nay là bắt đầu thông báo. Thì nó là có mở bài, có thân bài, có kết thúc thì nó là câu chuyện. Nhưng không phải thông tin em quá hư hỏng.

[111.4s - 113.1s] Khả Như: quá trời quá đất cái miệng rồi

[115.0s - 123.3s] Khả Như: Có nghĩa là vì chương trình này, nhờ chương trình này, em mới được Tiến với Hữu Đằng, anh Tiến mới được gặp em. Mới số đầu còn nhẹ nhàng, số sau là em rồi.

[125.2s - 131.7s] Khả
```

#### 🏆 [TỪ MÔ HÌNH ĐÃ GẮN NÃO PEFT]:
*(Nó sẽ đóng khuôn cứng ngắc các Header #)* 
```text
# 1. TỔNG QUAN CUỘC HỌP

Cuộc họp/Chương trình Chuyện Họ, Chuyện Mình kéo dài 3509.3s, bao gồm 14 người nói. Nội dung chính bao gồm các thảo luận, chia sẻ từ các diễn giả và thảo luận từ khán giả. Nội dung thảo luận bao gồm các chủ đề như: Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình, Chuyện Họ, Chuyện Mình
```

---
