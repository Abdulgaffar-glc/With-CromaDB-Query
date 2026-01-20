# With-CromaDB-Query
Tamamdır, anladım. Kod bloğu içine hapsetmeden, doğrudan burada okunabilir bir formatta, Türkçe olarak ve projenin detaylarını tam yansıtacak şekilde yazıyorum.

Aşağıdaki metni kopyalayıp GitHub'daki README.md dosyana yapıştırabilirsin.
🔍 ChromaDB ile Video İçi Anlamsal Arama ve Özetleme

Bu proje, video içeriklerini analiz etmek, anlamlı ana kareleri (keyframe) çıkarmak ve doğal dil sorguları kullanarak video içinde arama yapmak için geliştirilmiş yapay zeka tabanlı bir akıştır (pipeline).

Videodaki sahneleri anlamak için Facebook DINOv2, görüntüleri metne dökmek için Salesforce BLIP-2 ve verileri vektörel olarak saklayıp aramak için ChromaDB teknolojilerini kullanır.
🚀 Öne Çıkan Özellikler

    Akıllı Kare Yakalama: Videodaki bulanık veya anlamsız kareleri otomatik olarak eler.

    Görsel Embedding (DINOv2): dinov2_vitl14 (Large) modeli ile karelerin yüksek kaliteli vektör temsillerini çıkarır.

    Dinamik Kümeleme (K-Means): Benzer kareleri gruplar ve video süresine göre dinamik olarak "Keyframe" (Anahtar Kare) sayısını belirler.

    Otomatik Açıklama (BLIP-2): Seçilen her anahtar kare için yapay zeka ile İngilizce açıklama metni üretir.

    Vektör Veritabanı (ChromaDB): Oluşturulan açıklamaları vektör uzayında saklar.

    Anlamsal Arama: Kullanıcının yazdığı metni (örn: "kavga ediyorlar", "kırmızı araba") analiz eder ve videodaki en alakalı anı saniyesiyle birlikte bulur.

🛠️ Kullanılan Teknolojiler

Bu proje aşağıdaki güçlü kütüphane ve modelleri bir araya getirir:

    OpenCV: Video işleme ve görüntü netlik kontrolü.

    PyTorch: Derin öğrenme modellerinin çalıştırılması.

    DINOv2 (Meta AI): Görsel özellik çıkarımı için Vision Transformer modeli.

    BLIP-2 (Salesforce): Görselden metin üretimi (Image Captioning) için multimodal model.

    K-Means (Scikit-Learn): Görüntüleri sahne bazlı gruplamak için kümeleme algoritması.

    Sentence-Transformers: Metinleri vektöre dönüştürmek için.

    ChromaDB: Açık kaynaklı vektör veritabanı.

⚙️ Kurulum

Projenin çalışması için Python 3.8+ ve tercihen bir NVIDIA GPU gereklidir (Modeller büyük olduğu için CPU yavaş kalabilir).

Gerekli kütüphaneleri yükleyin:

pip install opencv-python torch numpy scikit-learn pillow transformers chromadb sentence-transformers
💻 Kullanım

    Analiz etmek istediğiniz videoyu proje klasörüne atın (Kod içinde varsayılan olarak /content/video.mp4 yolu tanımlıdır, bunu kendi dosya yolunuza göre düzenleyebilirsiniz).

    Python kodunu çalıştırın.

    Kod tamamlandığında, script'in sonundaki query_text değişkenine yazdığınız cümleyi video içinde arayacak ve sonuçları getirecektir.

Örnek Arama Sorgusu:

query_text = "two cars racing"
📊 Nasıl Çalışır? (Adım Adım)

    Ön İşleme: Video taranır, her 30 karede bir örnek alınır ve bulanık kareler (Laplacian yöntemi ile) elenir.

    Vektörleştirme: Geçerli kareler DINOv2 modeline sokulur ve sayısal vektörlere dönüştürülür.

    Sahne Seçimi: K-Means algoritması bu vektörleri kümeler. Her kümenin merkezindeki kare "temsilci kare" olarak seçilir.

    Metne Dökme: Seçilen kareler BLIP-2 modeline verilir ve o karede ne olduğunu anlatan bir metin üretilir.

    İndeksleme: Üretilen metinler ve zaman damgası (timestamp) bilgileri ChromaDB'ye kaydedilir.

    Arama: Kullanıcı bir cümle yazdığında, bu cümle vektöre çevrilir ve veritabanındaki en yakın sahnelerle eşleştirilir.
