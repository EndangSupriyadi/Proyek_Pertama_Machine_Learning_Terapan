# Laporan Proyek Machine Learning - Endang Supriyadi

## Domain Proyek
Kondisi ekonomi dan keuangan merupakan salah satu topik yang selalu berkembang dan menjadi perhatian masyarakat, baik masyarakat indonesia maupun masyarakat dunia. Perkembangan ekonomi tersebut akan senantiasa berubah seiring naiknya kebutuhan pokok, perubahan tersebut disebabkan oleh inflasi yang menurunkan daya beli masyarakat terhadap barang atau jasa akibat nilai tukar mata uang yang menurun. Emas /Gold merupakan barang yang berharga saat ini,selain sebagai perhiasan emas juga sebagai investasi. Peran investasi sangat penting untuk dapat mencegah penurunan lebih rendah lagi atau bahkan dapat meningkatkan pertumbuhan ekonomi [1]. Tujuan dari sebuah investasi adalah sebuah keuntungan maka banyak orang berinvestasi dengan emas, karena harga emas atau logam mulia cenderung stabil dan beresko rendah dibandingkan dengan investasi lain. Ketika terjadi inflasi harga emas tidak berubah dan cenderung aman [2]. Dengan teknologi machine learning memprediksi harga emas bisa dengan cepat karena cukup dengan dilatih dengan dataset yang sudah bersih sehingga bisa meningkatkan akurasi dalam memprediksi harga emas.

Kontribusi dari penerapan machine learning dalam memprediksi harga emas adalah memberikan perkiraan yang lebih akurat dan cepat dibandingkan dengan metode tradisional. Dengan menggunakan teknik machine learning, dapat memanfaatkan pola kompleks dalam data historis untuk membuat prediksi yang lebih baik tentang pergerakan harga emas di masa depan, yang dapat membantu investor dan pelaku pasar membuat keputusan investasi yang lebih baik.



## _Business Understanding_
1. _Problem Statements_
- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap harga emas?
- Berapa harga pasar emas dengan karakteristik atau fitur tertentu?
2. _Goals_
- Mengetahui fitur yang paling berkorelasi dengan harga emas yang akan digunakan untuk pelatihan model dan nantinya akan menghasilkan prediksi yang akurat terkait dengan harga emas untuk melakukan investasi.
- Membuat model machine learning yang dapat memprediksi harga emas seakurat mungkin berdasarkan fitur-fitur yang ada dengan membangun model regresi dan menggunakan metrik Mean Squared Error (MSE) atau Root Mean Square Error (RMSE) untuk mengukur seberapa jauh hasil prediksi dengan nilai yang sebenarnya. 


## _Data Understanding_

Data historis yang diambil dari Yahoo Finance untuk Gold ETF memiliki 7 kolom: Tanggal, Open, High, Low, Close, Adjusted Close, dan Volume. Berikut penjelasan mengenai variabel data tersebut:
- Tanggal (Date): Ini adalah tanggal perdagangan untuk setiap data historis.
- Open: Harga pembukaan Gold ETF pada hari tersebut.
- High: Harga tertinggi Gold ETF yang dicapai pada hari tersebut.
- Low: Harga terendah Gold ETF yang dicapai pada hari tersebut.
- Close: Harga penutupan Gold ETF pada hari tersebut.
- Adjusted Close: Harga penutupan yang telah disesuaikan dengan faktor-faktor seperti dividen, pemecahan saham (stock split), dan penerbitan saham baru. Adjusted Close dianggap sebagai representasi harga yang lebih akurat untuk analisis jangka panjang.
- Volume: Jumlah lembar saham Gold ETF yang diperdagangkan pada hari tersebut.

dataset ada 1718 rows dan 81 columns

hal menarik data close dan adjusted close sama nilainya maka dari itu diperbolehkan memilih salah satu saja

sumber dataset [https://www.kaggle.com/datasets/sid321axn/gold-price-prediction-dataset/data]

### Eksploratory Data
membaca dataset
dengan arakan path yang ada datasetnya dan tuliskan nama datasetnya sesuai script dibawah, setalah itu membaca dan menampilkan datanya seperti tabel 1. data ini masih bersifat mentah belum dibersihkan dan di filter fitur apa saja yang akan digunakan.
```
# load the dataset
df = '/content/gdrive/MyDrive/Kaggle/FINAL_USO.csv'
golds = pd.read_csv(df)
golds
```
tabel 1 <br>
<a href="https://ibb.co/jT5CkJp"><img src="https://i.ibb.co/1LnCGzS/Screenshot-2024-03-21-185708.png" alt="Screenshot-2024-03-21-185708" border="0"></a>
 <br>
Menampilkan info DataFrame dari dataset
di gambar 1 menampilkan typedata yang nantinya sebagai acuan  kedepannya 
```
golds.info()
```
<br>
gambar 1 <br>
<a href="https://imgbb.com/"><img src="https://i.ibb.co/x5sSZTq/Screenshot-2024-03-21-224103.png" alt="Screenshot-2024-03-21-224103" border="0"></a><br /><a target='_blank' href='https://imgbb.com/'>new logo png hd</a><br />
<br>
menampilkan hasil statistik dari dataframe seperti count, mean dll

``` 
golds.describe()
```
<br>
gambar 2 <br>
<a href="https://ibb.co/G0gz16s"><img src="https://i.ibb.co/rmnDLjb/Screenshot-2024-03-23-045627.png" alt="Screenshot-2024-03-23-045627" border="0"></a>
<br>

### Data Preparation
<br>
melakukan transformasi pada data sehingga menjadi bentuk yang cocok untuk proses pemodelan

Cek Nilai _Missing Value_
missing value merupakan nilai yang tidak ada atau NaNN yang ada di dataset. missing value bisa mempengaruhi kualiatas prediksi model sehingga harus dihapus atau ganti dengan nilai mean, count, dll. lalu mencek nilai _missing value_ dari kolom open, high dan low.

```
open = (golds.Open == 0).sum()
high = (golds.High == 0).sum()
low = (golds.Low == 0).sum()

print("Nilai 0 di kolom open ada: ", open)
print("Nilai 0 di kolom high ada: ", high)
print("Nilai 0 di kolom low ada: ", low)
```
data open, high, low nilai massing valuenya bernilai 0 jadi dalam fitur itu tidak ada nilai missing value 

#### Mengatasi outliers dengan IQR
yaitu untuk mengidentifikasi outlier yang berada diluar Q1 dan Q3. nilai apapun yang berada diluar batas ini dianggap sebagai outlier
visualisasi boxplot pada kolom open di gambar 3 akan terlihat apakah ada nilai outliers bisa dilihat dari lingkaran yang berjarak
<br>
```
sns.boxplot(x=golds['Open'])
```
<br>
gambar 3 <br>
<a href="https://ibb.co/377TDsd"><img src="https://i.ibb.co/wRRd9Lg/Screenshot-2024-03-23-000757.png" alt="Screenshot-2024-03-23-000757" border="0"></a>
<br>
visualisasi boxplot pada kolom high di gambar 4 akan terlihat apakah ada nilai outliers bisa dilihat dari lingkaran yang berjarak 
<br>

```
sns.boxplot(x=golds['High'])
``` 
<br>
gambar 4 <br>
<a href="https://imgbb.com/"><img src="https://i.ibb.co/pyQhFsw/Screenshot-2024-03-23-001031.png" alt="Screenshot-2024-03-23-001031" border="0"></a>
<br>
visualisasi boxplot pada kolom low di gambar 5 akan terlihat apakah ada nilai outliers bisa dilihat dari lingkaran yang berjarak  <br>

```
sns.boxplot(x=golds['Low'])
```
<br>
gambar 5 <br>
<a href="https://ibb.co/hB70Pv5"><img src="https://i.ibb.co/gSjY5C1/Screenshot-2024-03-23-001128.png" alt="Screenshot-2024-03-23-001128" border="0"></a>
<br>

Untuk mengatasi outliers gunakan metode IQR. metode IQR digunakan untuk mengidentifikasi outlier yang berada di luar Q1 dan Q3. Nilai apa pun yang berada di luar batas ini dianggap sebagai outlier. 
<br>
persamaan IQR : 
<br>
Batas bawah = Q1 - 1.5 * IQR

Batas atas = Q3 + 1.5 * IQR
<br>
```
Q1 = golds.quantile(0.25)
Q3 = golds.quantile(0.75)
IQR=Q3-Q1
golds=golds[~((golds<(Q1-1.5*IQR))|(golds>(Q3+1.5*IQR))).any(axis=1)]

# Cek ukuran dataset setelah kita drop outliers
golds.shape

#output
(835, 80)

```
menghitung korelasi antara kolom-kolom dalam dataframe goals dan menvisualisasikannya sehingga jika semakin tinggi nilai korelasi semakin kuat hubungan antara kolom target dan kolom yang bersangkutan. di gambar 6 korelasi yang kuat itu kolom high, low, open, close dan adj close. adj close dan close nilainya sama maka boleh pilih salah satu saja. di bawah ini menngunakan target columnnya itu close

```
target_column = 'Close'

# Calculate correlation matrix
correlation_matrix = golds.corr()
correlations = correlation_matrix[target_column]
plt.figure(figsize=(15, 20))
sns.barplot(x=correlations.values, y=correlations.index)
plt.title(f'Correlation with {target_column}')
plt.xlabel('correlations')
plt.ylabel('Columns')
plt.xticks(rotation=45)
plt.show()
```
gambar 6 
<br>
<a href="https://ibb.co/QbLKxgv"><img src="https://i.ibb.co/mDkcPj5/download-13.png" alt="download-13" border="0"></a>
 <br>


##### Train Test Split
membagi data latih dan data uji 80:20, proporsi tersebut sangat umum digunakan.
tujuannya agar data uji yang berperan sebagai data baru tidak terkotori dengan informasi yang didapatkan dari data latih.
```
X = golds.drop(["Close"],axis =1)
y = golds["Close"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')

#Output
Total # of sample in whole dataset: 835
Total # of sample in train dataset: 668
Total # of sample in test dataset: 167
```

#### Standarisasi
adalah teknik transformasi yang paling umum digunakan dalam tahap persiapan pemodelan. untuk fitur numerik tidak akan melakukan transformasi dengan one-hot-encoding seperti pada fitur kategori. tapi akan menggunakan teknik StandarScaler dari library Scikitlearn
StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi.  StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1.
```
numerical_features = ['Open','High', 'Low']
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head() <br>

```
gambar 7 
<br>
<a href="https://imgbb.com/"><img src="https://i.ibb.co/895YWpB/Screenshot-2024-03-21-224324.png" alt="Screenshot-2024-03-21-224324" border="0"></a>
<br>
```
X_train[numerical_features].describe().round(4)
```
gambar 8
<br>

<a href="https://imgbb.com/"><img src="https://i.ibb.co/Jj6WRDv/Screenshot-2024-03-21-224401.png" alt="Screenshot-2024-03-21-224401" border="0"></a>
<br>

### Modeling

Pengembangan model akan menggunakan beberapa algoritma machine learning yaitu _K-Nearest Neighbor, Random Forest, dan Boosting Algorithm._ Dari ketiga model ini, akan dipilih satu model yang memiliki nilai kesalahan prediksi terkecil. Dengan kata lain, dengan membuat model seakurat mungkin, yaitu model dengan nilai kesalahan sekecil mungkin.
```
# Siapkan dataframe untuk analisis model
models = pd.DataFrame(index=['train_mse', 'test_mse'], columns=['KNN', 'RandomForest', 'Boosting'])
```
#### Model KNN 
algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. untuk menentukan titik mana dalam data yang paling mirip dengan input baru, KNN menggunakan perhitungan ukuran jarak. metrik ukuran jarak yang juga sering dipakai antara lain: Euclidean distance dan Manhattan distance. Sebagai contoh, jarak Euclidean dihitung sebagai akar kuadrat dari jumlah selisih kuadrat antara titik a dan titik b. Dirumuskan sebagai berikut: <br>


$$ d(x,y) = { \sqrt{ \left( \sum_{n=1}^n (xi-yi)^2 \right) }}$$ 
<br>


#### Model Random Forest
boosting, algoritma ini bertujuan untuk meningkatkan performa atau akurasi prediksi. Caranya adalah dengan menggabungkan beberapa model sederhana dan dianggap lemah (weak learners) sehingga membentuk suatu model yang kuat (strong ensemble learner). Ada dua teknik pendekatan dalam membuat model ensemble, yaitu bagging dan boosting. Jangan bingung dulu dengan istilah ini ya. Kita akan bahas satu per satu. Bagging atau bootstrap aggregating adalah teknik yang melatih model dengan sampel random. Dalam teknik bagging, sejumlah model dilatih dengan teknik sampling with replacement (proses sampling dengan penggantian)


#### Model Boosting Algorithm
boosting, algoritma ini bertujuan untuk meningkatkan performa atau akurasi prediksi. Caranya adalah dengan menggabungkan beberapa model sederhana dan dianggap lemah (weak learners) sehingga membentuk suatu model yang kuat (strong ensemble learner). Algoritma boosting muncul dari gagasan mengenai apakah algoritma yang sederhana seperti linear regression dan decision tree dapat dimodifikasi untuk dapat meningkatkan performa. Algoritma yang menggunakan teknik boosting bekerja dengan membangun model dari data latih. Kemudian ia membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan. 


### _Evaluation_
Metrik digunakan untuk mengevaluasi seberapa baik model dalam memprediksi harga. Untuk kasus regresi, beberapa metrik yang biasanya digunakan adalah Mean Squared Error (MSE) atau Root Mean Square Error (RMSE). Secara umum, metrik ini mengukur seberapa jauh hasil prediksi dengan nilai yang sebenarnya. <br>
Rumus MSE:
<br>

$$ MSE = { 1/N {  \sum_{i=1}^n (yi-ypred_i)^2  }}$$ 

<br>
sebelum menghitung nilai MSE, perlu melakukan proses scaling fitur numerik pada data uji. Sebelumnya, melakukan proses scaling pada data latih untuk menghindari kebocoran data. Sekarang, setelah model selesai dilatih dengan 3 algoritma, yaitu KNN, Random Forest, dan Adaboost, perlu melakukan proses scaling terhadap data uji. Hal ini harus dilakukan agar skala antara data latih dan data uji sama dan bisa melakukan evaluasi.



```
# Lakukan scaling terhadap fitur numerik pada X_test sehingga memiliki rata-rata=0 dan varians=1
X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])
# Buat variabel mse yang isinya adalah dataframe nilai mse data train dan test pada masing-masing algoritma
mse = pd.DataFrame(columns=['train', 'test'], index=['KNN','RF','Boosting'])

# Buat dictionary untuk setiap algoritma yang digunakan
model_dict = {'KNN': knn, 'RF': RF, 'Boosting': boosting}

# Hitung Mean Squared Error masing-masing algoritma pada data train dan test
for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))/1e3
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))/1e3

# Panggil mse
Mean Squared Error (MSE) metrik ini mengukur seberapa jauh hasil prediksi dengan nilai yang sebenarnya. 
```
tabel 2
<br>
<a href="https://imgbb.com/"><img src="https://i.ibb.co/Lx9dsm3/Screenshot-2024-03-21-224441.png" alt="Screenshot-2024-03-21-224441" border="0"></a>
<br>


Dari gambar tabel 2, terlihat bahwa, model Random Forest (RF) memberikan nilai eror yang paling kecil. Sedangkan model dengan KNN memiliki eror yang paling besar. Sehingga model RF yang akan yang pilih sebagai model terbaik untuk melakukan prediksi harga diamonds.

dalam tabel 3 nilai prediksi Random Forest mendekati nilai uji walaupun nilai prediksi model Boasting juga mendekati nilai uji

```
prediksi = X_test.iloc[:1].copy()
pred_dict = {'y_true':y_test[:1]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)

pd.DataFrame(pred_dict)
```
tabel 3 <br>

<a href="https://imgbb.com/"><img src="https://i.ibb.co/mHq9FTD/Screenshot-2024-03-21-224627.png" alt="Screenshot-2024-03-21-224627" border="0"></a>
<br>
Terlihat bahwa prediksi dengan Random Forest (RF) memberikan hasil yang paling mendekati nilai uji

Referensi Jurnal : <br>
[1] M. D. H. Mela Priantika, Sari Wulandari, “Harga Emas Terhadap Minat Nasabah Berinvestasi Menggunakan Produk Tabungan Emas,” J. Penelit. Pendidik. Sos. Hum., vol. 6, no. 1, pp. 8–12, 2021, doi: 10.32696/jp2sh.v6i1.714. <br>
[2]	M. Muharrom, “Analisis Komparasi Algoritma Data Mining Naive Bayes, K-Nearest Neighbors dan Regresi Linier Dalam Prediksi Harga Emas,” Bull. Inf. Technol., vol. 4, no. 4, pp. 430–438, 2023, doi: 10.47065/bit.v4i4.986.

link [https://jurnal-lp2m.umnaw.ac.id/index.php/JP2SH/article/view/714/518] <br>
link [https://journal.fkpt.org/index.php/BIT/article/view/986/509]
