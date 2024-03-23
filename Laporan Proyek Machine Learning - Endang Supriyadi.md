# Laporan Proyek Machine Learning - Endang Supriyadi

## Domain Proyek
Kondisi ekonomi dan keuangan merupakan salah satu topik yang selalu berkembang dan menjadi perhatian masyarakat, baik masyarakat indonesia maupun masyarakat dunia. Perkembangan ekonomi tersebut akan senantiasa berubah seiring naiknya kebutuhan pokok, perubahan tersebut disebabkan oleh inflasi yang menurunkan daya beli masyarakat terhadap barang atau jasa akibat nilai tukar mata uang yang menurun. Emas /_Gold_ merupakan barang yang berharga saat ini,selain sebagai perhiasan emas juga sebagai investasi. Peran investasi sangat penting untuk dapat mencegah penurunan lebih rendah lagi atau bahkan dapat meningkatkan pertumbuhan ekonomi [1]. Tujuan dari sebuah investasi adalah sebuah keuntungan maka banyak orang berinvestasi dengan emas, karena harga emas atau logam mulia cenderung stabil dan beresko rendah dibandingkan dengan investasi lain. Ketika terjadi inflasi harga emas tidak berubah dan cenderung aman [2]. Dengan teknologi machine learning memprediksi harga emas bisa dengan cepat karena cukup dengan dilatih dengan dataset yang sudah bersih sehingga bisa meningkatkan akurasi dalam memprediksi harga emas.

Kontribusi dari penerapan _machine learning_ dalam memprediksi harga emas adalah memberikan perkiraan yang lebih akurat dan cepat dibandingkan dengan metode tradisional. Dengan menggunakan teknik _machine learning_, dapat memanfaatkan pola kompleks dalam data historis untuk membuat prediksi yang lebih baik tentang pergerakan harga emas di masa depan, yang dapat membantu investor dan pelaku pasar membuat keputusan investasi yang lebih baik.



## _Business Understanding_
1. _Problem Statements_
- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap harga emas?
- Berapa harga pasar emas dengan karakteristik atau fitur tertentu?
2. _Goals_
- Mengetahui fitur yang paling berkorelasi dengan harga emas yang akan digunakan untuk pelatihan model dan nantinya akan menghasilkan prediksi yang akurat terkait dengan harga emas untuk melakukan investasi.
- Membuat model machine learning yang dapat memprediksi harga emas seakurat mungkin berdasarkan fitur-fitur yang ada dengan membangun model regresi dan menggunakan metrik _Mean Squared Error_ (MSE) atau _Root Mean Square Error_ (RMSE) untuk mengukur seberapa jauh hasil prediksi dengan nilai yang sebenarnya. 


## _Data Understanding_

Data historis yang diambil dari Yahoo Finance untuk _Gold ETF_ memiliki 7 kolom: Tanggal, Open, High, Low, Close, Adjusted Close, dan Volume. Berikut penjelasan mengenai variabel data tersebut:
- Tanggal (Date): Ini adalah tanggal perdagangan untuk setiap data historis.
- Open: Harga pembukaan Gold ETF pada hari tersebut.
- High: Harga tertinggi Gold ETF yang dicapai pada hari tersebut.
- Low: Harga terendah Gold ETF yang dicapai pada hari tersebut.
- Close: Harga penutupan Gold ETF pada hari tersebut.
- Adjusted Close: Harga penutupan yang telah disesuaikan dengan faktor-faktor seperti dividen, pemecahan saham (stock split), dan penerbitan saham baru. Adjusted Close dianggap sebagai representasi harga yang lebih akurat untuk analisis jangka panjang.
- Volume: Jumlah lembar saham Gold ETF yang diperdagangkan pada hari tersebut.

dataset ada 1718 rows dan 81 columns

hal menarik data close dan adjusted close sama nilainya maka dari itu diperbolehkan memilih salah satu saja <br>

menghitung korelasi antara kolom-kolom dalam dataframe golds dan menvisualisasikannya sehingga jika semakin tinggi nilai korelasi semakin kuat hubungan antara kolom target dan kolom yang bersangkutan. di gambar 1 korelasi yang kuat itu kolom high, low, open, close dan adj close. adj close dan close nilainya sama maka boleh pilih salah satu saja. di bawah ini menngunakan target columnnya itu close


gambar 1
<br>
<a href="https://ibb.co/QbLKxgv"><img src="https://i.ibb.co/mDkcPj5/download-13.png" alt="download-13" border="0"></a>
 <br>


sumber dataset [https://www.kaggle.com/datasets/sid321axn/gold-price-prediction-dataset/data]

### _Eksploratory Data_
membaca dataset
pertama arahkan alamat path penyimpanan dataset , setalah itu membaca dan menampilkan data dengan "read_csv" pastikan dataset berformat csv. Data ini masih bersifat mentah perlu menyaring data apa saja yang dibutuhkan dalam mengerjakan proyek ini seperti pada Tabel 1.

Tabel 1 <br>

| | **Date** | **Open** | **High** | **Low** | **Close** |
|-|----------|----------|----------|---------|-----------|
|0|2011-12-15|154.740005|154.949997|151.710007|152.330002|
|1|2011-12-16|154.309998|155.369995|153.899994|155.229996|
|2|2011-12-19|155.479996|155.860001|154.360001|154.869995|
|3|2011-12-20|156.820007|157.429993|156.580002|156.979996|
|4|2011-12-21|156.979996|157.529999|156.130005|157.160004|





<br>
Menampilkan info DataFrame dari dataset
di tabel 2 menampilkan typedata dengan perintah " golds.info()" yang nantinya sebagai acuan kedepannya 

<br>
Tabel 2 <br>


|**#**|**Column**        |**Non-Null Count**|**Dtype**|
|-----|------------------|------------------|---------|
| 0   |Date              |1718 non-null     |object   |
| 1   |Open              |1718 non-null     |float64  |
| 2   |High              |1718 non-null     |float64  |
| 3   |Low               |1718 non-null     |float64  |
| 4   |Close             |1718 non-null     |float64  |
<br>

pada gambar 4 dengan perintah "golds.describe()" menampilkan hasil statistik dari dataframe seperti count, mean dll. hal ini agar mengetahui masing masing statistik perkategorinya 

<br>
Tabel 3 <br>


|       | **Open**  | **High**  | **Low**    | **Close** |
|-------|-----------|-----------|------------|-----------|
|count  |1718.000000|1718.000000|1718.000000	|1718.000000|
|mean   |127.323434 |127.854237	|126.777695	 |127.319482	|
|std    |17.526993  |17.631189	 |17.396513   |17.536269  |
|min    |100.919998 |100.989998 |100.230003  |100.500000 |
|25%	   |116.220001 |116.540001	|115.739998  |116.052502	|
|50%	   |121.915001 |122.325001	|121.369999  |121.795002	|
|75%	   |128.427494 |129.087498 |127.840001	 |128.470001	|
|max	   |173.199997 |174.070007	|172.919998	 |173.610001	|

<br>
penjelasan : <br>
Count  adalah jumlah sampel pada data. <br>
Mean adalah nilai rata-rata. <br>
Std adalah standar deviasi. <br>
Min yaitu nilai minimum setiap kolom. <br>
25% adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama. <br>
50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah). <br>
75% adalah kuartil ketiga.<br>
Max adalah nilai maksimum.<br>
<br>

#### _Univariate Analysis_

Dalam Dataset ini berisikan time series setiap harga emas per tanggalnya. sehingga perlu diubah type data dari date yang tadinya object menjadi date agar bisa digunakan dan divisualisasikan dengan harga close emas.
<br>
gambar 2
<br>

<a href="https://ibb.co/s96FRML"><img src="https://i.ibb.co/PG95jfB/Screenshot-2024-03-23-123452.png" alt="Screenshot-2024-03-23-123452" border="0"></a>
<br>


### _Data Preparation_
<br>
melakukan transformasi pada data sehingga menjadi bentuk yang cocok untuk proses pemodelan

Cek Nilai _Missing Value_
dengan" .sum() akan menampilkan data _missing value_. _missing value_ merupakan nilai yang tidak ada atau NaNN yang ada di dataset. _missing value_ bisa mempengaruhi kualiatas prediksi model sehingga harus dihapus atau ganti dengan nilai mean, count, dll. lalu mencek nilai _missing value_ dari kolom open, high dan low.

#### Mengatasi _outliers_ dengan IQR
yaitu untuk mengidentifikasi _outlier_ yang berada diluar Q1 dan Q3. nilai apapun yang berada diluar batas ini dianggap sebagai _outlier_ dengan perintah "sns.boxplot()" akan menampilkan visualisasi boxplot. boxplot terlihat apakah ada nilai outliers bisa dilihat dari lingkaran yang berjarak. 

visualisasi boxplot pada kolom open di gambar 3 terlihat ada lingkaran yang berjarak
<br>
gambar 3 <br>
<a href="https://ibb.co/377TDsd"><img src="https://i.ibb.co/wRRd9Lg/Screenshot-2024-03-23-000757.png" alt="Screenshot-2024-03-23-000757" border="0"></a>
<br>
visualisasi boxplot pada kolom high di gambar 4 terlihat ada lingkaran yang berjarak
<br>
gambar 4 <br>
<a href="https://imgbb.com/"><img src="https://i.ibb.co/pyQhFsw/Screenshot-2024-03-23-001031.png" alt="Screenshot-2024-03-23-001031" border="0"></a>
<br>
visualisasi boxplot pada kolom low di gambar 5 terlihat ada lingkaran yang berjarak
<br>
gambar 5 <br>
<a href="https://ibb.co/hB70Pv5"><img src="https://i.ibb.co/gSjY5C1/Screenshot-2024-03-23-001128.png" alt="Screenshot-2024-03-23-001128" border="0"></a>
<br>

Untuk mengatasi _outliers_ gunakan metode IQR. metode IQR digunakan untuk mengidentifikasi _outlier_ yang berada di luar Q1 dan Q3. Nilai apa pun yang berada di luar batas ini dianggap sebagai outlier. 
<br>
persamaan IQR : 
<br>
Batas bawah = Q1 - 1.5 * IQR
Batas atas = Q3 + 1.5 * IQR

jika melebihi batas tersebut maka akan dihapus terlihat pada ouput shape yang berkurang
<br>



##### Train Test Split
membagi data latih dan data uji 80:20, proporsi tersebut sangat umum digunakan.
tujuannya agar data uji yang berperan sebagai data baru tidak terkotori dengan informasi yang didapatkan dari data latih. data set ini berubah mejadi  835 data untuk jumlah dataset, 668 data untuk latih, dan 167 data untuk uji karena menggunakan perbandingan 80 data latih dan 20 data uji.


#### Standarisasi
adalah teknik transformasi yang paling umum digunakan dalam tahap persiapan pemodelan. untuk fitur numerik tidak akan melakukan transformasi dengan one-hot-encoding seperti pada fitur kategori. tapi akan menggunakan teknik StandarScaler dari library Scikitlearn
StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi.  StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1. seperti Tabel 4 dan Tabel 5
<br>
Tabel 4 
<br>

|     |**Open**   |**High**   |**Low**    |
|-----|-----------|-----------|-----------|
|946  |-2.165675	 |-2.204928	 |-2.194981  |
|1381 |0.421708	  |0.373206	  |0.449826   |
|893  |-1.680881	 |-1.676567	 |-1.634293  |
|1044 |-0.293227	 |-0.245926	 |-0.250350  |
|762  |-0.406255	 |-0.446433	 |-0.465053  |
<br>

Tabel 5
<br>



|     |**Open**   |**High**   |**Low**    |
|-----|-----------|-----------|-----------|
|count|668.0000 	 |668.0000		 |668.0000	  |
|mean |-0.0000		  |-0.0000		  |-0.0000	   |
|std  |1.0007	  	 |1.0007	  	 |1.0007	    |
|min  |-2.5279 		 |-2.5667	 	 |-2.5697    |
|25%	 |-0.5213	 	 |-0.5281	 	 |-0.5515    |
|50%	 |0.0949   	 |0.1077   	 |0.1025     |
|75%	 |0.7363   	 |0.7366	  	 |0.7281     |
|max	 |3.1112   	 |3.0855   	 |3.0933     |

<br>

### _Modeling_

Pengembangan model akan menggunakan beberapa algoritma machine learning yaitu _K-Nearest Neighbor, Random Forest, dan Boosting Algorithm._ Dari ketiga model ini, akan dipilih satu model yang memiliki nilai kesalahan prediksi terkecil. Dengan kata lain, dengan membuat model seakurat mungkin, yaitu model dengan nilai kesalahan sekecil mungkin.

#### Model _KNN_
algoritma _KNN_ menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. untuk menentukan titik mana dalam data yang paling mirip dengan input baru, _KNN_ menggunakan perhitungan ukuran jarak. metrik ukuran jarak yang juga sering dipakai antara lain: _Euclidean distance_ dan _Manhattan distance_. Sebagai contoh, jarak _Euclidean_ dihitung sebagai akar kuadrat dari jumlah selisih kuadrat antara titik a dan titik b. Dirumuskan sebagai berikut: <br>


$$ d(x,y) = { \sqrt{ \left( \sum_{n=1}^n (xi-yi)^2 \right) }}$$ 
<br>
Parameter yang digunakan :
- n_neighbors: Jumlah tetangga yang akan digunakan untuk membuat prediksi.
- weights: Cara memberi bobot pada tetangga. Misalnya, "uniform" memberi bobot yang sama pada semua tetangga, sementara "distance" memberi bobot yang lebih besar pada tetangga yang lebih dekat.
- metric: Metrik jarak yang digunakan untuk mengukur kedekatan antara titik data.

#### Model _Random Forest_
boosting, algoritma ini bertujuan untuk meningkatkan performa atau akurasi prediksi. Caranya adalah dengan menggabungkan beberapa model sederhana dan dianggap lemah (weak learners) sehingga membentuk suatu model yang kuat (strong ensemble learner). Ada dua teknik pendekatan dalam membuat model ensemble, yaitu bagging dan boosting. Bagging atau bootstrap aggregating adalah teknik yang melatih model dengan sampel random. Dalam teknik bagging, sejumlah model dilatih dengan teknik sampling with replacement (proses sampling dengan penggantian)

Parameter yang digunakan :
- n_estimators: Jumlah pohon keputusan yang akan dibangun dalam ensemble.
- max_depth: Kedalaman maksimum setiap pohon keputusan dalam ensemble.
- min_samples_split: Jumlah sampel minimum yang diperlukan untuk membagi node dalam pohon.
- min_samples_leaf: Jumlah sampel minimum yang diperlukan di leaf node dalam pohon.

#### Model _Boosting Algorithm_
boosting, algoritma ini bertujuan untuk meningkatkan performa atau akurasi prediksi. Caranya adalah dengan menggabungkan beberapa model sederhana dan dianggap lemah (_weak learners_) sehingga membentuk suatu model yang kuat (_strong ensemble learner_). Algoritma boosting muncul dari gagasan mengenai apakah algoritma yang sederhana seperti linear regression dan decision tree dapat dimodifikasi untuk dapat meningkatkan performa. Algoritma yang menggunakan teknik boosting bekerja dengan membangun model dari data latih. Kemudian ia membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan. 

Parameter yang digunakan :
- n_estimators: Jumlah pohon keputusan yang akan dibangun dalam _ensemble_.
- learning_rate: Tingkat pembelajaran (learning rate) yang mengontrol kontribusi setiap pohon terhadap model.
- max_depth: Kedalaman maksimum setiap pohon keputusan dalam _ensemble_.
- subsample: Fraksi sampel yang akan digunakan untuk pelatihan setiap pohon.
- colsample_bytree: Fraksi fitur yang akan digunakan untuk pelatihan setiap pohon.
  
dengan membandingkan ketiga model itu untuk mengetahui model mana yang lebih akurat dalam menangani kasus ini dengan menggunakan metrik mse bisa mengentahui seberapa besar error dari model ketiga itu

Alasan menggunakan tiga algoritma yaitu _KNN_, _Random Forest_ dan _Boosting_ : <br>
1. _KNN_ adalah algoritma yang sederhana dan intuitif, serta mudah dipahami. Ini dapat digunakan untuk melakukan regresi (misalnya, memprediksi harga emas berikutnya berdasarkan data historis).
2. _Random Forest_ adalah metode _ensemble_ yang kuat yang terdiri dari sejumlah besar pohon keputusan. Hal ini sangat efektif dalam menangani _overfitting_, menangani data yang tidak seimbang, serta dalam menangani data dengan jumlah fitur yang besar.
3. _Boosting_ adalah teknik ensemble yang memperbaiki kinerja model dengan cara berurutan memperbaiki kesalahan prediksi model sebelumnya. Model ini biasanya memberikan kinerja yang sangat baik dalam berbagai masalah prediksi.

### _Evaluation_
Metrik digunakan untuk mengevaluasi seberapa baik model dalam memprediksi harga. Untuk kasus regresi, beberapa metrik yang biasanya digunakan adalah _Mean Squared Error_ (MSE) atau _Root Mean Square Error_ (RMSE). Secara umum, metrik ini mengukur seberapa jauh hasil prediksi dengan nilai yang sebenarnya. <br>
Rumus MSE:
<br>

$$ MSE = { 1/N {  \sum_{i=1}^n (yi-ypred_i)^2  }}$$ 

<br>
sebelum menghitung nilai MSE, perlu melakukan proses scaling fitur numerik pada data uji. Sebelumnya, melakukan proses scaling pada data latih untuk menghindari kebocoran data. Sekarang, setelah model selesai dilatih dengan 3 algoritma, yaitu _KNN_, _Random Forest_, dan _Boosting_, perlu melakukan proses scaling terhadap data uji. Hal ini harus dilakukan agar skala antara data latih dan data uji sama dan bisa melakukan evaluasi. 

Dari gambar gambar 11, menggunakan MSE untuk melihat seberapa kemungkinan besar errornya. 
1. Model _KNN_ hasilnya untuk data latih 0.029005 dan uji 0.034733
2. Model _Random Forest_ untuk data latih 0.000008 dan uji 0.000046
3. Model _Boosting_ untuk data latih 0.000575 dan uji 0.000602
terlihat bahwa, model _Random Forest_ (RF) memberikan nilai eror yang paling kecil yaitu 0.000008 untuk data latih dan 0.000046 untuk data uji. Sedangkan model dengan _KNN_ memiliki eror yang paling besar yaitu 0.029005 untuk data latih dan  0.034733 untuk data uji. Sehingga dapat dilihat model _Random Forest_ sebagai model terbaik untuk melakukan prediksi harga golds. 
<br>
Tabel 6
<br>


|              |**train**   |**test**   |
|--------------|------------|-----------|
|**KNN**       |0.029005	 	 |0.034733		 |
|**RF**        |0.000008		  |0.000046		 |
|**Boosting**  |0.000575	  	|0.000602	  |
<br>


dalam Tabel 7 terdapat nilai uji 109.139999	dengan hasil prediksi KNN : 117.9	, prediksi RF : 109.3	 dan, prediksi Boosting : 107.8. nilai prediksi _Random Forest_ (RF) mendekati nilai uji walaupun nilai prediksi model _Boasting_ juga mendekati nilai uji tapi yang lebih mendekati itu model _Random Forest_ dengan perbedaan sekitar 0.2 saja.

Tabel 7 <br>




|     |**y_true**   |**prediksi_KNN**|**prediksi_RF**|**prediksi_Boosting**|
|-----|-------------|----------------|---------------|---------------------|
|915  |109.139999 	 |117.9   		      |109.3   	      |          107.8      |
<br>

Proyek ini berhasil karena mendapatkan solusi dari masalah yang ingin diselesaikan dengan mencoba fitur mana saja yang dapat digunakan dalam proyek ini dan mengetahui model mana yang bisa lebih akurat dalam memprediksi harga emas. dengan Model Random Forest bisa lebih akurat dalam memprediksi harga emas karena nilai dari MSE nya lebih kecil dari pada KNN dan Boosting.

Kesimpulannya <br>
1. fitur yang paling berpengaruh yaitu fitur close dan adj close karena memiliki korelasi yang sangat kuat terhadao fitur open , high , dan low.
2. setelah membanding ke tiga model itu menggunakan MSE dan melakukan uji, dihasilkan bahwa model Random Forest memiliki nilai error yang rendah dan ketika diuji nilai prediksinya hampir mendekati dibanding dengan model lainnya .

Referensi Jurnal : <br>
[1] M. D. H. Mela Priantika, Sari Wulandari, “Harga Emas Terhadap Minat Nasabah Berinvestasi Menggunakan Produk Tabungan Emas,” J. Penelit. Pendidik. Sos. Hum., vol. 6, no. 1, pp. 8–12, 2021, doi: 10.32696/jp2sh.v6i1.714. <br>
[2]	M. Muharrom, “Analisis Komparasi Algoritma Data Mining Naive Bayes, K-Nearest Neighbors dan Regresi Linier Dalam Prediksi Harga Emas,” Bull. Inf. Technol., vol. 4, no. 4, pp. 430–438, 2023, doi: 10.47065/bit.v4i4.986.

link [https://jurnal-lp2m.umnaw.ac.id/index.php/JP2SH/article/view/714/518] <br>
link [https://journal.fkpt.org/index.php/BIT/article/view/986/509]
