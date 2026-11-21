import random

random.seed(42)
dataset = []
for _ in range(300):
    jaringan = random.choice([1, 0])
    jam = random.randint(0, 23)
    sinyal = random.randint(-95, -45)
    perangkat = random.randint(1, 15)
    dasar = 55.0 if jaringan == 1 else 30.0
    sibuk = -12.0 if (19 <= jam <= 22) else 0.0
    kecepatan = dasar + (sinyal + 100) * 1.1 + sibuk + (perangkat * -1.4) + random.gauss(0, 3)
    dataset.append((jaringan, jam, sinyal, perangkat, round(max(1.5, min(150.0, kecepatan)), 2)))

latih, uji = dataset[:240], dataset[240:]
bobot, bias, lr = [0.0, 0.0, 0.0, 0.0], 0.0, 0.0001

for _ in range(5000):
    for j, jm, s, p, y in latih:
        fitur = [j, jm, s + 100, p]
        pred = bias + sum(w * x for w, x in zip(bobot, fitur))
        err = pred - y
        bias -= lr * err
        for i in range(4):
            bobot[i] -= lr * err * fitur[i]

def tebak(j, jm, s, p):
    hasil = bias + sum(w * x for w, x in zip(bobot, [j, jm, s + 100, p]))
    return max(1.5, min(150.0, hasil))

mae = sum(abs(y - tebak(j, jm, s, p)) for j, jm, s, p, y in uji) / len(uji)
print("=== HASIL PREDIKSI KECEPATAN INTERNET ===")
print(f"Rata-rata Error (MAE): {mae:.2f} Mbps")
print(f"Prediksi WiFi (Jam 09:00, -52 dBm, 2 user)    : {tebak(1, 9, -52, 2):.2f} Mbps")
print(f"Prediksi Seluler (Jam 20:00, -82 dBm, 8 user) : {tebak(0, 20, -82, 8):.2f} Mbps")
