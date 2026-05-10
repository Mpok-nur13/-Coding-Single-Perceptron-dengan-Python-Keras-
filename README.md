# 🧠 Machine Learning dengan 2 Input — Logika AND & Regresi Linear

Notebook ini memperkenalkan konsep **neural network dengan dua input** menggunakan Keras, melalui dua studi kasus: **Gerbang Logika AND** dan **Regresi Linear dengan dua bobot**.

---

## 📌 Deskripsi

Berbeda dari model satu input, notebook ini menggunakan `input_shape=[2]` — artinya model menerima dua nilai sekaligus untuk membuat prediksi. Dua masalah diangkat:

1. **Gerbang AND** — model belajar mengenali pola logika biner.
2. **Regresi Linear 2 Variabel** — model belajar menemukan bobot `w1` dan `w2` dari persamaan `y = w1*x1 + w2*x2`.

---

## 📁 Struktur Notebook

---

### 🔷 Bagian 1 — Gerbang Logika AND

#### Cell 1 — Tabel Logika AND

```
x1  x2  y
1   1   1
1   0   0
0   1   0
0   0   0
```

Gerbang AND menghasilkan output `1` hanya jika **kedua input bernilai 1**, selainnya menghasilkan `0`.

---

#### Cell 2 — Membuat & Kompilasi Model

```python
import keras
import numpy as np

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[2])])
model.compile(optimizer='sgd', loss='mean_squared_error')
```

Model menggunakan satu neuron dengan **dua input**, dikompilasi menggunakan optimizer SGD dan loss MSE — sama seperti sebelumnya namun kini menerima dua fitur sekaligus.

---

#### Cell 3 — Set Data Latih

```python
xs = np.array([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=int)
ys = np.array([1, 0, 0, 0], dtype=int)
```

Seluruh kombinasi input gerbang AND dimasukkan sebagai data latih (4 baris, 2 kolom).

---

#### Cell 4 — Ringkasan Arsitektur Model

```python
model.summary()
```

Menampilkan struktur model: jumlah layer, jumlah neuron, dan total parameter yang akan dilatih.

---

#### Cell 5 — Cek Bobot Awal

```python
weights = model.get_weights()
```

Bobot model diinisialisasi secara **acak** sebelum pelatihan. Cell ini digunakan untuk melihat nilai awal bobot `w1`, `w2`, dan bias `b`.

---

#### Cell 6 — Training

```python
model.fit(xs, ys, epochs=1000)
```

Model dilatih selama **1000 epoch** hingga mampu mempelajari pola AND dari data.

---

#### Cell 7 — Prediksi

```python
data = [1, 1]
answer = model.predict([data])
print(answer)  # Diharapkan mendekati 1
```

Menguji model dengan input `[1, 1]` yang seharusnya menghasilkan nilai mendekati `1`.

---

#### Cell 8 — Cek Bobot Akhir

```python
weights = model.get_weights()
```

Melihat bobot final setelah pelatihan — nilai `w1`, `w2`, dan `b` yang sudah disesuaikan oleh model untuk merepresentasikan logika AND.

---

### 🔷 Bagian 2 — Regresi Linear 2 Variabel

#### Cell 9 — Tabel Data

```
Persamaan: y = 2*x1 + 4*x2

x1  x2  y
2   3   16
4   1   12
5   4   28
...
```

Model **tidak diberitahu** bahwa `w1=2` dan `w2=4`. Ia harus menemukannya sendiri dari 12 pasang data latih.

---

#### Cell 10 — Model, Kompilasi & Data

```python
model2 = keras.Sequential([keras.layers.Dense(units=1, input_shape=[2])])
model2.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([[2,3],[4,1],[5,4],[7,5],[8,2],[2,1],
               [4,9],[8,2],[7,1],[6,5],[1,1],[3,2]], dtype=int)
ys = np.array([16,12,28,34,24,8,44,24,18,32,6,14], dtype=int)
```

Model kedua (`model2`) dikonfigurasi identik dengan model pertama namun dilatih pada data regresi linear.

---

#### Cell 11 — Cek Bobot Awal

```python
weights = model2.get_weights()
```

Bobot awal yang masih acak sebelum proses pelatihan.

---

#### Cell 12 — Training

```python
model2.fit(xs, ys, epochs=1000)
```

Model dilatih 1000 epoch untuk menemukan nilai `w1` dan `w2` yang mendekati 2 dan 4.

---

#### Cell 13 — Prediksi

```python
data = [1, 3]
answer = model2.predict([data])
print(answer)
# Seharusnya: 1*2 + 3*4 = 14
```

Menguji model — jika berhasil belajar, output akan mendekati `14`.

---

#### Cell 14 — Cek Bobot Akhir

```python
weights = model2.get_weights()
```

Bobot akhir diharapkan mendekati `w1 ≈ 2` dan `w2 ≈ 4`, membuktikan model berhasil menemukan pola dari data.

---

## 🔍 Konsep Penting

| Konsep | Penjelasan |
|---|---|
| `input_shape=[2]` | Model menerima 2 nilai input secara bersamaan |
| `get_weights()` | Mengambil nilai bobot dan bias dari model |
| `model.summary()` | Menampilkan ringkasan arsitektur model |
| Bobot awal | Diinisialisasi acak, lalu dioptimasi saat training |
| Bobot akhir | Nilai yang dipelajari model untuk merepresentasikan pola data |

---

## 📊 Perbandingan Dua Studi Kasus

| Aspek | Gerbang AND | Regresi Linear |
|---|---|---|
| Tujuan | Klasifikasi biner (0 atau 1) | Prediksi nilai kontinu |
| Data latih | 4 baris (semua kombinasi AND) | 12 baris |
| Output yang diharapkan | Mendekati 0 atau 1 | Mendekati nilai `y = 2x1 + 4x2` |
| Bobot yang dipelajari | Representasi logika AND | Mendekati `w1=2`, `w2=4` |

---

## ⚙️ Requirements

```
tensorflow / keras
numpy
```

Install dengan:

```bash
pip install tensorflow numpy
```

---

## 🚀 Cara Menjalankan

1. Buka notebook di [Google Colab](https://colab.research.google.com/) (disarankan GPU T4).
2. Jalankan cell secara berurutan dari atas ke bawah.
3. Perhatikan perubahan nilai loss di setiap epoch — semakin kecil berarti model semakin akurat.
4. Bandingkan bobot awal dan bobot akhir untuk melihat proses belajar model.

---

## 💡 Insight

Notebook ini menunjukkan bahwa neural network sederhana (1 neuron) sudah mampu **mempelajari hubungan linear antar dua variabel** — baik untuk masalah klasifikasi sederhana (AND) maupun regresi. Kunci utamanya ada di **bobot (`weights`)** yang terus diperbarui selama training hingga model menemukan pola yang tepat dari data.
