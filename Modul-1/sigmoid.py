import numpy as np
import matplotlib.pyplot as plt


# 1. FUNGSI CRISP

def crisp_memuaskan(rating, threshold=4.0):
    """
    Crisp:
    rating >= 4  -> 1
    rating < 4   -> 0
    """
    return np.where(rating >= threshold, 1.0, 0.0)


# 2. FUNGSI FUZZY LINEAR NAIK

def fuzzy_linear(rating, a=2.5, b=4.5):
    """
    Fuzzy Linear Naik:
    x < 2.5       -> 0
    2.5 <= x <= 4.5 -> (x-2.5)/(4.5-2.5)
    x > 4.5       -> 1
    """
    derajat = (rating - a) / (b - a)

    return np.clip(derajat, 0.0, 1.0)


# 3. FUNGSI FUZZY SIGMOID

def fuzzy_sigmoid(rating, x0=3.5, k=2.0):
    """
    Fungsi keanggotaan Sigmoid.

    x0 = titik tengah
    k  = kecuraman lereng
    """
    return 1 / (1 + np.exp(-k * (rating - x0)))

# 4. MEMBUAT DOMAIN RATING

ratings = np.linspace(1.0, 5.0, 500)


# 5. MENGHITUNG NILAI KEANGGOTAAN

y_crisp = crisp_memuaskan(ratings)
y_linear = fuzzy_linear(ratings)
y_sigmoid = fuzzy_sigmoid(ratings)


# 6. MENAMPILKAN HASIL DI TERMINAL

print("=" * 75)
print("PERBANDINGAN CRISP, FUZZY LINEAR, DAN FUZZY SIGMOID")
print("=" * 75)

print(
    f"{'Rating':<10}"
    f"{'Crisp':<12}"
    f"{'Fuzzy Linear':<18}"
    f"{'Fuzzy Sigmoid':<18}"
)

print("-" * 75)


# Nilai yang akan diuji
nilai_uji = [
    1.0,
    2.0,
    2.5,
    3.0,
    3.5,
    3.8,
    3.9,
    4.0,
    4.01,
    4.2,
    4.5,
    5.0
]


for x in nilai_uji:

    crisp = float(crisp_memuaskan(x))
    linear = float(fuzzy_linear(x))
    sigmoid = float(fuzzy_sigmoid(x))

    print(
        f"{x:<10.2f}"
        f"{crisp:<12.3f}"
        f"{linear:<18.3f}"
        f"{sigmoid:<18.3f}"
    )


print("=" * 75)

print("\nParameter Sigmoid:")
print("x0 = 3.5")
print("k  = 2.0")


# 7. MEMBUAT GRAFIK

plt.figure(figsize=(10, 6))


# Kurva Crisp
plt.step(
    ratings,
    y_crisp,
    label="Crisp (Threshold = 4.0)",
    linewidth=2.5,
    where="post"
)


# Kurva Fuzzy Linear
plt.plot(
    ratings,
    y_linear,
    label="Fuzzy Linear (2.5 - 4.5)",
    linewidth=2.5
)


# Kurva Fuzzy Sigmoid
plt.plot(
    ratings,
    y_sigmoid,
    label="Fuzzy Sigmoid (x0=3.5, k=2.0)",
    linewidth=2.5
)


# Garis titik tengah Sigmoid
plt.axvline(
    x=3.5,
    linestyle="--",
    alpha=0.6,
    label="x0 = 3.5"
)


# 8. PENGATURAN GRAFIK

plt.title(
    "Perbandingan Crisp, Fuzzy Linear, dan Fuzzy Sigmoid",
    fontsize=13,
    fontweight="bold"
)

plt.xlabel(
    "Rating Pengguna (Skala 1 - 5)",
    fontsize=11
)

plt.ylabel(
    "Derajat Keanggotaan",
    fontsize=11
)

plt.xlim(1, 5)
plt.ylim(-0.05, 1.05)

plt.grid(
    True,
    linestyle=":",
    alpha=0.6
)

plt.legend()

plt.tight_layout()

# 9. SIMPAN GRAFIK

plt.savefig(
    "perbandingan_crisp_linear_sigmoid.png",
    dpi=300
)

print("\nGrafik berhasil disimpan sebagai:")
print("perbandingan_crisp_linear_sigmoid.png")

# 10. TAMPILKAN GRAFIK

plt.show()