# Laporan Proyek Machine Learning - Endang Supriyadi

## Domain Proyek
Kondisi ekonomi dan keuangan merupakan salah satu topik yang selalu berkembang dan menjadi perhatian masyarakat, baik masyarakat indonesia maupun masyarakat dunia. Perkembangan ekonomi tersebut akan senantiasa berubah seiring naiknya kebutuhan pokok, perubahan tersebut disebabkan oleh inflasi yang menurunkan daya beli masyarakat terhadap barang atau jasa akibat nilai tukar mata uang yang menurun. Emas /Gold merupakan barang yang berharga saat ini,selain sebagai perhiasan emas juga sebagai investasi. Peran investasi sangat penting untuk dapat mencegah penurunan lebih rendah lagi atau bahkan dapat meningkatkan pertumbuhan ekonomi [1]. Tujuan dari sebuah investasi adalah sebuah keuntungan maka banyak orang berinvestasi dengan emas, karena harga emas atau logam mulia cenderung stabil dan beresko rendah dibandingkan dengan investasi lain. Ketika terjadi inflasi harga emas tidak berubah dan cenderung aman [2]. maka dari itu tujuan dari proyek ini yaitu menprediksi harga emas dengan machine learning dengan beberapa tahapan diantaranya data collection: mengumpulkan data histori harga emas, data processing : mencek apakah data set yang diambil sudah bersih atau masih kotor, melakukan normalisasi fitur, dan pemilihan fitur. Model training : disini menggunakan 3 model machine learning KNN, Random Forest dan Boosting karena agar tau mana model yang efesien untuk proyek ini. Evaluasi Model : Mencek hasil prediksi dari ketiga model itu dan mana prediksi yang lebih akurat

Referensi Jurnal : <br>
[1] M. D. H. Mela Priantika, Sari Wulandari, “Harga Emas Terhadap Minat Nasabah Berinvestasi Menggunakan Produk Tabungan Emas,” J. Penelit. Pendidik. Sos. Hum., vol. 6, no. 1, pp. 8–12, 2021, doi: 10.32696/jp2sh.v6i1.714. <br>
[2]	M. Muharrom, “Analisis Komparasi Algoritma Data Mining Naive Bayes, K-Nearest Neighbors dan Regresi Linier Dalam Prediksi Harga Emas,” Bull. Inf. Technol., vol. 4, no. 4, pp. 430–438, 2023, doi: 10.47065/bit.v4i4.986.


link [https://journal.fkpt.org/index.php/BIT/article/view/986/509]

## Business Understanding
1. Problem Statements
- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap harga emas?
- Berapa harga pasar emas dengan karakteristik atau fitur tertentu?
2. Goals
- Mengetahui fitur yang paling berkorelasi dengan harga emas yang akan digunakan untuk pelatihan model dan nantinya akan menghasilkan prediksi yang akurat terkait dengan harga emas untuk melakukan investasi.
- Membuat model machine learning yang dapat memprediksi harga emas seakurat mungkin berdasarkan fitur-fitur yang ada dengan membangun model regresi dan menggunakan metrik Mean Squared Error (MSE) atau Root Mean Square Error (RMSE) untuk mengukur seberapa jauh hasil prediksi dengan nilai yang sebenarnya. ini akan bahas lebih detail mengenai metrik ini di modul Evaluasi. Pengembangan model akan menggunakan beberapa algoritma machine learning yaitu K-Nearest Neighbor, Random Forest, dan Boosting Algorithm. Dari ketiga model ini, akan dipilih satu model yang memiliki nilai kesalahan prediksi terkecil. Dengan kata lain, kita akan membuat model seakurat mungkin, yaitu model dengan nilai kesalahan sekecil mungkin.


## Data Understanding

Data historis yang diambil dari Yahoo Finance untuk Gold ETF memiliki 7 kolom: Tanggal, Open, High, Low, Close, Adjusted Close, dan Volume. Berikut penjelasan mengenai variabel data tersebut:
- Tanggal (Date): Ini adalah tanggal perdagangan untuk setiap data historis.
- Open: Harga pembukaan Gold ETF pada hari tersebut.
- High: Harga tertinggi Gold ETF yang dicapai pada hari tersebut.
- Low: Harga terendah Gold ETF yang dicapai pada hari tersebut.
- Close: Harga penutupan Gold ETF pada hari tersebut.
- Adjusted Close: Harga penutupan yang telah disesuaikan dengan faktor-faktor seperti dividen, pemecahan saham (stock split), dan penerbitan saham baru. Adjusted Close dianggap sebagai representasi harga yang lebih akurat untuk analisis jangka panjang.
- Volume: Jumlah lembar saham Gold ETF yang diperdagangkan pada hari tersebut.

dataset ada 1718 rows dan 81 columns

sumber dataset [https://www.kaggle.com/datasets/sid321axn/gold-price-prediction-dataset/data]
#### Library
``` 
! pip install kaggle
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
```
#### Download Dataset
``` 
os.environ['KAGGLE_CONFIG_DIR'] = "/content/gdrive/My Drive/Kaggle"
%cd /content/gdrive/My Drive/Kaggle
!kaggle datasets download -d sid321axn/gold-price-prediction-dataset
# unzip dataset
!unzip \*.zip && rm *.zip.
```


### Eksploratory Data
membaca dataset
```
# load the dataset
df = '/content/gdrive/MyDrive/Kaggle/FINAL_USO.csv'
golds = pd.read_csv(df)
golds
```
<img width="886" alt="Screenshot 2024-03-21 185708" src="https://github.com/EndangSupriyadi/Proyek_Pertama_Machine_Learning_Terapan/assets/103325979/461c74cd-b871-4c86-abc2-647a961e7d5c"> <br>
Menampilkan info DataFrame dari dataset
disini kita bisa melihat typedata yang nantinya sebagai acuan kita kedepannya
```
golds.info()
```
<br>
<img width="242" alt="Screenshot 2024-03-21 224103" src="https://github.com/EndangSupriyadi/Proyek_Pertama_Machine_Learning_Terapan/assets/103325979/8b131c60-244c-4f65-9628-2dbfe8c6daff"> <br>
menampilkan hasil statistik dari dataframe seperti count, mean dll
``` 
golds.describe()
```
Cek Nilai Missing Value 
jika ada maka kita harus atau gunakan nilai mean, median, atau nilai yang lain sesuai ketentuannya

```
open = (golds.Open == 0).sum()
high = (golds.High == 0).sum()
low = (golds.Low == 0).sum()

print("Nilai 0 di kolom open ada: ", open)
print("Nilai 0 di kolom high ada: ", high)
print("Nilai 0 di kolom low ada: ", low)
```
data diatas tidak ada missing value

#### Mengatasi outliers dengan IQR
yaitu untuk mengidentifikasi outlier yang berada diluar Q1 dan Q3. nilai apapun yang berada diluar batas ini dianggap sebagai outlier
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
menghitung korelasi antara kolom-kolom dalam dataframe goals dan menvisualisasikannya sehingga jika semakin tinggi nilai korelasi semakin kuat hubungan antara kolom target dan kolom yang bersangkutan. 

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

![download (13)](https://github.com/EndangSupriyadi/Proyek_Pertama_Machine_Learning_Terapan/assets/103325979/f2b98eac-8797-48df-ac6e-f967b548ed25) <br>

### Data Preparation
disini kita melakukan transformasi pada data sehingga menjadi bentuk yang cocok untuk proses pemodelan
##### Train Test Split
membagi data latih dan data uji 80:20, proporsi tersebut sangat umum digunakan.
tujuannya agar data uji yang berperan sebagai data baru tidak terkotori dengan informasi yang kita dapatkan dari data latih.
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
adalah teknik transformasi yang paling umum digunakan dalam tahap persiapan pemodelan. untuk fitur numerik kita tidak akan melakukan transformasi dengan one-hot-encoding seperti pada fitur kategori. kita akan menggunakan teknik StandarScaler dari library Scikitlearn
StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi.  StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1.
```
numerical_features = ['Open','High', 'Low']
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()
```
<img width="197" alt="Screenshot 2024-03-21 224324" src="https://github.com/EndangSupriyadi/Proyek_Pertama_Machine_Learning_Terapan/assets/103325979/9a12223d-1810-48da-a64b-2ad2719aa286"> <br>
```
X_train[numerical_features].describe().round(4)
```
<img width="176" alt="Screenshot 2024-03-21 224401" src="https://github.com/EndangSupriyadi/Proyek_Pertama_Machine_Learning_Terapan/assets/103325979/f475b4cd-457f-49b3-ab8d-7fedf5696c9a">
<br>

### Modeling

mecoba membuat 3 buah model machine learning dengan algoritma : 
1. K-Nearest Neighbor (KNN)
2. Random Forest
3. Boosting Algorithm 
yang nantinya kita akan membandingkan mana model yang efektif dalam menyelesaikan kasus ini
```
# Siapkan dataframe untuk analisis model
models = pd.DataFrame(index=['train_mse', 'test_mse'], columns=['KNN', 'RandomForest', 'Boosting'])
```
#### Model KNN 
algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. 

menggunakan nilai K =10
```

knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)

models.loc['train_mse','knn'] = mean_squared_error(y_pred = knn.predict(X_train), y_true=y_train)
```
#### Model Random Forest
termasuk model kategori ensemble(group) learning yaitu model prediksi yang terdiri dari beberapa model dan bekerjasama, sehingga tidak keberhasilan akan lebih tinggi dibandingkan model yang bekerja sendiri.

menggunakan nilai n_etimator (jumlah trees)=45
max_depth (panjang atau kedalam pohon) = 16
random_state =60
n_jobs =-1

```
# buat model prediksi
RF = RandomForestRegressor(n_estimators=45, max_depth=16, random_state=60, n_jobs=-1)
RF.fit(X_train, y_train)

models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)
```

#### Model Boosting Algorithm
boosting, algoritma ini bertujuan untuk meningkatkan performa atau akurasi prediksi. Caranya adalah dengan menggabungkan beberapa model sederhana dan dianggap lemah (weak learners) sehingga membentuk suatu model yang kuat (strong ensemble learner). Algoritma boosting muncul dari gagasan mengenai apakah algoritma yang sederhana seperti linear regression dan decision tree dapat dimodifikasi untuk dapat meningkatkan performa. 

nilai learning_rate = 0.05
random_state = 60

```
boosting = AdaBoostRegressor(learning_rate=0.05, random_state=60)
boosting.fit(X_train, y_train)
models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train), y_true=y_train)
```

### Evaluation
dengan Metrik MSE

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
mse
```
<img width="163" alt="Screenshot 2024-03-21 224441" src="https://github.com/EndangSupriyadi/Proyek_Pertama_Machine_Learning_Terapan/assets/103325979/5a7ef6f9-9ead-437d-95c4-c73dfec478ab">
<br>
nilai error yang paling kecil yaitu random forest

disini nilai prediksi Random Forest mendekati nilai uji walaupun nilai prediksi model Boasting juga mendekati

```
prediksi = X_test.iloc[:1].copy()
pred_dict = {'y_true':y_test[:1]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)

pd.DataFrame(pred_dict)
```
<img width="289" alt="Screenshot 2024-03-21 224627" src="https://github.com/EndangSupriyadi/Proyek_Pertama_Machine_Learning_Terapan/assets/103325979/04bafd8d-178b-4d99-877a-a66fcbcaac42"> <br>
Terlihat bahwa prediksi dengan Random Forest (RF) memberikan hasil yang paling mendekati.
