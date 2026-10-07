import os
import struct
import zlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fftpack import dct
from scipy.stats import chi2
from scipy.ndimage import median_filter

# =====================================================================
# STI-KONFIGURASJON
# =====================================================================
BASE_DIR = r"C:\dev"

COVER_PATH = os.path.join(BASE_DIR, "bilde.jpg")
SECRET_PATH = os.path.join(BASE_DIR, "bilde2.jpg")
STEGO_OUTPUT = os.path.join(BASE_DIR, "stego_output.png")
RESTORED_SECRET_OUTPUT = os.path.join(BASE_DIR, "restored_secret.png")
RESTORED_COVER_OUTPUT = os.path.join(BASE_DIR, "restored_cover_exact.png")
ANALYSIS_OUTPUT = os.path.join(BASE_DIR, "bildeanalyse_oversikt.png")

SECRET_KEY = 4281900
TARGET_SECRET_SIZE = (128, 128)
PCA_COMPONENTS = 16  # Kutter payload med ~85% samtidig som bildekarakter bevares

# =====================================================================
# 1. PCA MODELLERING & KRYPTERING
# =====================================================================

def compress_with_pca(image: np.ndarray, n_components: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dekomponerer bildet via SVD på sentrerte pikselrader."""
    X = image.astype(np.float32)
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    components = Vt[:n_components, :]             # (k, w)
    projected = np.dot(X_centered, components.T)  # (h, k)
    return projected, components, mean

def reconstruct_from_pca(projected: np.ndarray, components: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Gjenoppbygger tilnærmet bilde fra PCA-koeffisientene."""
    recon = np.dot(projected, components) + mean
    return np.clip(np.round(recon), 0, 255).astype(np.uint8)

def encrypt_pca_payload(projected: np.ndarray, components: np.ndarray, mean: np.ndarray, key: int) -> bytes:
    """Kvantiserer, pakker og krypterer PCA-matrisene."""
    h, k = projected.shape
    _, w = components.shape

    p_min, p_max = float(projected.min()), float(projected.max())
    c_min, c_max = float(components.min()), float(components.max())

    q_proj = np.clip(np.round((projected - p_min) / (p_max - p_min + 1e-8) * 255), 0, 255).astype(np.uint8)
    q_comp = np.clip(np.round((components - c_min) / (c_max - c_min + 1e-8) * 255), 0, 255).astype(np.uint8)
    q_mean = np.clip(np.round(mean), 0, 255).astype(np.uint8)

    raw_payload = q_proj.tobytes() + q_comp.tobytes() + q_mean.tobytes()
    n = len(raw_payload)

    rng = np.random.RandomState(key)
    flat_data = np.frombuffer(raw_payload, dtype=np.uint8)
    perm = rng.permutation(n)
    keystream = rng.randint(0, 256, size=n, dtype=np.uint8)
    encrypted = np.bitwise_xor(flat_data[perm], keystream)

    header = struct.pack(">HHHHffffI", h, w, k, 0, p_min, p_max, c_min, c_max, n)
    return header + encrypted.tobytes()

def decrypt_pca_payload(payload: bytes, key: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dekrypterer og dekvantiserer PCA-representasjonen."""
    header = payload[:28]
    h, w, k, _, p_min, p_max, c_min, c_max, n = struct.unpack(">HHHHffffI", header)
    enc_data = np.frombuffer(payload[28:28 + n], dtype=np.uint8)

    rng = np.random.RandomState(key)
    perm = rng.permutation(n)
    keystream = rng.randint(0, 256, size=n, dtype=np.uint8)

    unmasked = np.bitwise_xor(enc_data, keystream)
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(n)
    raw = unmasked[inv_perm]

    proj_size = h * k
    comp_size = k * w

    q_proj = raw[:proj_size].reshape((h, k)).astype(np.float32)
    q_comp = raw[proj_size:proj_size + comp_size].reshape((k, w)).astype(np.float32)
    q_mean = raw[proj_size + comp_size:proj_size + comp_size + w].astype(np.float32)

    proj = (q_proj / 255.0) * (p_max - p_min) + p_min
    comp = (q_comp / 255.0) * (c_max - c_min) + c_min
    return proj, comp, q_mean

# =====================================================================
# 2. DIFFUS REVERSIBEL STEGANOGRAFI VIA DIFFERANSE-EKSPANSJON (DE)
# =====================================================================

def is_expandable(x: int, y: int, b: int) -> bool:
    """Verifiserer at ekspandert pikselpar forblir innenfor [0, 255]."""
    l = (x + y) // 2
    h = x - y
    h_prime = 2 * h + b
    x_prime = l + (h_prime + 1) // 2
    y_prime = l - h_prime // 2
    return 0 <= x_prime <= 255 and 0 <= y_prime <= 255

def embed_reversible_de(cover: np.ndarray, secret_payload: bytes, key: int) -> tuple[np.ndarray, int]:
    """Tapsfri reversibel embedding basert på differanse-ekspansjon og pseudotilfeldig spredning."""
    flat = cover.copy().flatten().astype(np.int32)
    num_pairs = len(flat) // 2

    x_vals = flat[0::2][:num_pairs]
    y_vals = flat[1::2][:num_pairs]

    rng = np.random.RandomState(key ^ 0xDEADBEEF)
    pair_order = rng.permutation(num_pairs)

    secret_bits = np.unpackbits(np.frombuffer(secret_payload, dtype=np.uint8))
    secret_bit_len = len(secret_bits)

    expandable_mask = np.zeros(num_pairs, dtype=bool)
    for p_idx in range(num_pairs):
        px = x_vals[p_idx]
        py = y_vals[p_idx]
        if is_expandable(px, py, 0) and is_expandable(px, py, 1):
            expandable_mask[p_idx] = True

    perm_expandable = [idx for idx in pair_order if expandable_mask[idx]]

    HEADER_BITS = 64
    total_payload_bits = HEADER_BITS + secret_bit_len

    if len(perm_expandable) < total_payload_bits:
        raise ValueError(
            f"Cover har kun {len(perm_expandable)} ekspanderbare par, men krever {total_payload_bits} bits."
        )

    header_bytes = struct.pack(">II", len(secret_payload), secret_bit_len)
    header_bits = np.unpackbits(np.frombuffer(header_bytes, dtype=np.uint8))
    all_bits = np.concatenate([header_bits, secret_bits])

    target_pairs = perm_expandable[:total_payload_bits]
    for i, p_idx in enumerate(target_pairs):
        px = x_vals[p_idx]
        py = y_vals[p_idx]
        b = int(all_bits[i])

        l = (px + py) // 2
        h = px - py
        h_prime = 2 * h + b

        x_vals[p_idx] = l + (h_prime + 1) // 2
        y_vals[p_idx] = l - h_prime // 2

    stego_flat = flat.copy()
    stego_flat[0::2][:num_pairs] = x_vals
    stego_flat[1::2][:num_pairs] = y_vals

    return np.clip(stego_flat, 0, 255).astype(np.uint8).reshape(cover.shape), total_payload_bits * 2

def extract_and_restore_de(stego: np.ndarray, key: int) -> tuple[bytes, np.ndarray]:
    """Trekker ut hemmelig payload og gjenoppretter coverbildet piksel-eksakt."""
    flat = stego.copy().flatten().astype(np.int32)
    num_pairs = len(flat) // 2

    x_vals = flat[0::2][:num_pairs]
    y_vals = flat[1::2][:num_pairs]

    rng = np.random.RandomState(key ^ 0xDEADBEEF)
    pair_order = rng.permutation(num_pairs)

    reversible_pairs = []
    for p_idx in pair_order:
        px = x_vals[p_idx]
        py = y_vals[p_idx]
        h_prime = px - py
        b = h_prime & 1
        h = (h_prime - b) // 2
        l = (px + py) // 2
        orig_x = l + (h + 1) // 2
        orig_y = l - h // 2
        if 0 <= orig_x <= 255 and 0 <= orig_y <= 255:
            reversible_pairs.append(p_idx)

    HEADER_BITS = 64
    extracted_bits = []
    for i in range(HEADER_BITS):
        p_idx = reversible_pairs[i]
        h_prime = x_vals[p_idx] - y_vals[p_idx]
        extracted_bits.append(h_prime & 1)

    header_bytes = np.packbits(np.array(extracted_bits, dtype=np.uint8)).tobytes()
    secret_byte_len, secret_bit_len = struct.unpack(">II", header_bytes)

    total_bits_to_extract = HEADER_BITS + secret_bit_len

    all_extracted_bits = []
    for i in range(total_bits_to_extract):
        p_idx = reversible_pairs[i]
        px = x_vals[p_idx]
        py = y_vals[p_idx]
        h_prime = px - py
        b = h_prime & 1
        all_extracted_bits.append(b)

        h = (h_prime - b) // 2
        l = (px + py) // 2
        x_vals[p_idx] = l + (h + 1) // 2
        y_vals[p_idx] = l - h // 2

    secret_raw_bits = np.array(all_extracted_bits[HEADER_BITS:HEADER_BITS + secret_bit_len], dtype=np.uint8)
    secret_payload = np.packbits(secret_raw_bits).tobytes()[:secret_byte_len]

    restored_flat = flat.copy()
    restored_flat[0::2][:num_pairs] = x_vals
    restored_flat[1::2][:num_pairs] = y_vals

    return secret_payload, np.clip(restored_flat, 0, 255).astype(np.uint8).reshape(stego.shape)

# =====================================================================
# 3. STATISTISKE OG SPEKTRALE ANALYSEVERKTØY & DETEKTOR
# =====================================================================

def get_pair_differences(img: np.ndarray) -> np.ndarray:
    flat = img.astype(np.int32).flatten()
    num_pairs = len(flat) // 2
    return flat[0::2][:num_pairs] - flat[1::2][:num_pairs]

def detect_difference_expansion(img: np.ndarray, threshold_imbalance: float = 0.035) -> dict:
    diffs = get_pair_differences(img)
    mask = (diffs >= -10) & (diffs <= 10)
    local_diffs = diffs[mask]

    even_count = np.sum((local_diffs % 2) == 0)
    odd_count = np.sum((local_diffs % 2) != 0)
    total_local = max(len(local_diffs), 1)

    imbalance = (even_count - odd_count) / total_local
    detected = imbalance < threshold_imbalance

    return {
        "detected": bool(detected),
        "imbalance": float(imbalance),
        "even_count": int(even_count),
        "odd_count": int(odd_count),
        "diffs": diffs,
        "konklusjon": "DE Steganografi Funnet!" if detected else "Rent bilde (Ingen DE funnet)"
    }

def compute_fft_spectrum(img: np.ndarray) -> np.ndarray:
    fshift = np.fft.fftshift(np.fft.fft2(img.astype(np.float64)))
    return np.log1p(np.abs(fshift))

def compute_block_dct_energy(img: np.ndarray, block_size: int = 8) -> np.ndarray:
    h, w = img.shape
    h_blocks, w_blocks = h // block_size, w // block_size
    energy_map = np.zeros((h_blocks, w_blocks), dtype=np.float64)

    def dct2(a):
        return dct(dct(a.T, norm='ortho').T, norm='ortho')

    hf_mask = np.zeros((block_size, block_size), dtype=bool)
    for r in range(block_size):
        for c in range(block_size):
            if r + c >= block_size:
                hf_mask[r, c] = True

    for i in range(h_blocks):
        for j in range(w_blocks):
            blk = img[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size].astype(np.float64)
            energy_map[i, j] = np.sum(np.abs(dct2(blk)[hf_mask]))

    return energy_map

def compute_noise_residual(img: np.ndarray) -> np.ndarray:
    filtered = median_filter(img, size=3)
    residual = np.abs(img.astype(np.int16) - filtered.astype(np.int16))
    return np.clip(residual * 8, 0, 255).astype(np.uint8)

def compute_chi_square(image: np.ndarray, block_size: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    flat = image.flatten()
    n = len(flat)
    x_pct, probs = [], []
    counts = np.zeros(256, dtype=np.int64)

    for i in range(0, n, block_size):
        blk = flat[i : i + block_size]
        u, c = np.unique(blk, return_counts=True)
        counts[u] += c

        pair_sums = counts[0::2] + counts[1::2]
        valid = pair_sums > 0
        k = np.sum(valid)

        if k <= 1:
            x_pct.append(((i + len(blk)) / n) * 100.0)
            probs.append(0.0)
            continue

        exp = pair_sums[valid] / 2.0
        chi_stat = np.sum(((counts[0::2][valid] - exp) ** 2) / exp) + \
                   np.sum(((counts[1::2][valid] - exp) ** 2) / exp)

        x_pct.append(((i + len(blk)) / n) * 100.0)
        probs.append(1.0 - (1.0 - chi2.cdf(chi_stat, k - 1)))

    return np.array(x_pct), np.array(probs)

# =====================================================================
# 4. HOVEDPROGRAM OG VISUALISERING
# =====================================================================

def main():
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR, exist_ok=True)

    if not os.path.exists(COVER_PATH):
        print(f"Genererer test-cover: {COVER_PATH}")
        y, x = np.mgrid[0:720, 0:1280]
        grad = np.clip((np.sin(x / 35.0) * 45 + np.cos(y / 35.0) * 45 + 128) + np.random.normal(0, 3, (720, 1280)), 0, 255)
        Image.fromarray(grad.astype(np.uint8)).save(COVER_PATH)

    if not os.path.exists(SECRET_PATH):
        print(f"Genererer hemmelig testbilde: {SECRET_PATH}")
        sec = np.zeros((128, 128), dtype=np.uint8)
        sec[32:96, 32:96] = 210
        Image.fromarray(sec).save(SECRET_PATH)

    print(f"Laster inn kildebilder fra {BASE_DIR}...")
    cover_img = np.array(Image.open(COVER_PATH).convert('L'), dtype=np.uint8)
    secret_img = np.array(Image.open(SECRET_PATH).convert('L').resize(TARGET_SECRET_SIZE, Image.Resampling.LANCZOS), dtype=np.uint8)

    # 1. PCA-dekomponering og kryptering
    print(f"Kjører PCA-dekomponering (k={PCA_COMPONENTS})...")
    proj, comp, mean_vec = compress_with_pca(secret_img, n_components=PCA_COMPONENTS)
    secret_payload = encrypt_pca_payload(proj, comp, mean_vec, key=SECRET_KEY)
    print(f"Kryptert PCA payload-størrelse: {len(secret_payload)} bytes")

    # 2. Diffus Differanse-Ekspansjon embedding
    print("Utfører diffus differanse-ekspansjon (DE)...")
    stego_img, total_pixels_mod = embed_reversible_de(cover_img, secret_payload, key=SECRET_KEY)
    Image.fromarray(stego_img).save(STEGO_OUTPUT)
    cap_pct = (total_pixels_mod / cover_img.size) * 100.0

    # 3. Kjører DE-steganalysedetektor
    print("\n--- KJØRER DIFFERANSE-EKSPANSJON DETEKSJON ---")
    det_clean = detect_difference_expansion(cover_img)
    det_stego = detect_difference_expansion(stego_img)

    print(f"Originalbilde: {det_clean['konklusjon']} (Partalls-overvekt: {det_clean['imbalance']:.4f})")
    print(f"Stego-bilde:   {det_stego['konklusjon']} (Partalls-overvekt: {det_stego['imbalance']:.4f})")
    print("----------------------------------------------\n")

    # 4. Ekstraksjon og full restaurering
    print("Henter ut data og gjenoppretter opprinnelig cover...")
    extracted_bytes, restored_cover = extract_and_restore_de(stego_img, key=SECRET_KEY)
    d_proj, d_comp, d_mean = decrypt_pca_payload(extracted_bytes, key=SECRET_KEY)
    restored_secret = reconstruct_from_pca(d_proj, d_comp, d_mean)

    Image.fromarray(restored_secret).save(RESTORED_SECRET_OUTPUT)
    Image.fromarray(restored_cover).save(RESTORED_COVER_OUTPUT)
    is_exact = np.array_equal(cover_img, restored_cover)
    print(f"Modifiserte piksler: {total_pixels_mod} ({cap_pct:.2f} % av bildet)")
    print(f"Eksakt reversert cover (lossless): {is_exact}")

    # 5. Signalanalyse
    print("Beregner Fourier-spekter, DCT-fordeling, støyavvik og chi-kvadrat...")
    lsb_clean = (cover_img & 1) * 255
    lsb_stego = (stego_img & 1) * 255
    fft_clean = compute_fft_spectrum(cover_img)
    fft_stego = compute_fft_spectrum(stego_img)
    dct_clean = compute_block_dct_energy(cover_img)
    dct_stego = compute_block_dct_energy(stego_img)
    res_clean = compute_noise_residual(cover_img)
    res_stego = compute_noise_residual(stego_img)
    x_c, p_c = compute_chi_square(cover_img)
    x_s, p_s = compute_chi_square(stego_img)

    # 6. Visualisering (3 rader x 5 kolonner for å inkludere statistisk bevisføring)
    fig, axes = plt.subplots(3, 5, figsize=(22, 13), constrained_layout=True)
    fig.suptitle(
        f"PCA-DE Steganografi & Deteksjonsstatistikk (Modifisert: {cap_pct:.2f} %) | Resultat: {det_stego['konklusjon']}",
        fontsize=14, fontweight='bold'
    )

    # RAD 1: Visuelle flater og LSB
    axes[0, 0].imshow(cover_img, cmap='gray')
    axes[0, 0].set_title("Originalt bilde", fontsize=10)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(stego_img, cmap='gray')
    axes[0, 1].set_title("Stego-bilde (PCA-DE)", fontsize=10)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(lsb_clean, cmap='gray')
    axes[0, 2].set_title("Original LSB", fontsize=10)
    axes[0, 2].axis('off')

    axes[0, 3].imshow(lsb_stego, cmap='gray')
    axes[0, 3].set_title("Stego LSB (Ingen støyblokker)", fontsize=10)
    axes[0, 3].axis('off')

    axes[0, 4].imshow(restored_secret, cmap='gray')
    axes[0, 4].set_title(f"Gjenopprettet Secret (k={PCA_COMPONENTS})", fontsize=10)
    axes[0, 4].axis('off')

    # RAD 2: Frekvens, DCT og Støyresidual
    axes[1, 0].imshow(fft_clean, cmap='viridis')
    axes[1, 0].set_title("FFT Spektrum (Original)", fontsize=10)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(fft_stego, cmap='viridis')
    axes[1, 1].set_title("FFT Spektrum (Stego)", fontsize=10)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(dct_clean, cmap='magma')
    axes[1, 2].set_title("8x8 DCT HF (Original)", fontsize=10)
    axes[1, 2].axis('off')

    axes[1, 3].imshow(dct_stego, cmap='magma')
    axes[1, 3].set_title("8x8 DCT HF (Stego)", fontsize=10)
    axes[1, 3].axis('off')

    axes[1, 4].imshow(res_stego, cmap='gray')
    axes[1, 4].set_title("Støyresidual (Stego)", fontsize=10)
    axes[1, 4].axis('off')

    # RAD 3: STATISTISKE BEVIS FOR DETEKSJONEN
    # Panel [2, 0]: Chi-kvadrat
    ax_chi = axes[2, 0]
    ax_chi.plot(x_c, p_c, label="Original", color="#1f77b4", lw=1.5)
    ax_chi.plot(x_s, p_s, label="Stego", color="#2ca02c", lw=1.5, linestyle="--")
    ax_chi.set_title(r"$\chi^2$-test (Blind for DE)", fontsize=10)
    ax_chi.set_xlabel("Flate (%)", fontsize=8)
    ax_chi.set_ylabel("Deteksjons-sannsynlighet", fontsize=8)
    ax_chi.set_ylim(-0.05, 1.05)
    ax_chi.legend(fontsize=8)
    ax_chi.grid(True, linestyle=":", alpha=0.5)

    # Panel [2, 1]: Differansehistogram (Selve kjernen i DE-avsløringen)
    ax_hist = axes[2, 1]
    bins = np.arange(-12, 13) - 0.5
    ax_hist.hist(det_clean["diffs"], bins=bins, density=True, alpha=0.45, color="#1f77b4", label="Original", edgecolor='black')
    ax_hist.hist(det_stego["diffs"], bins=bins, density=True, alpha=0.45, color="#d62728", label="Stego (Ekspandert)", edgecolor='black')
    ax_hist.set_title("Differansehistogram: h = x - y", fontsize=10, fontweight='bold')
    ax_hist.set_xlabel("Pikseldifferanse (h)", fontsize=8)
    ax_hist.set_ylabel("Tetthet", fontsize=8)
    ax_hist.set_xlim(-10, 10)
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, linestyle=":", alpha=0.5)

    # Panel [2, 2]: Paritetsfordeling (Partall vs Oddetall i nære differanser)
    ax_bars = axes[2, 2]
    cat_labels = ["Original", "Stego"]
    tot_clean = det_clean["even_count"] + det_clean["odd_count"]
    tot_stego = det_stego["even_count"] + det_stego["odd_count"]
    clean_ratios = [det_clean["even_count"] / tot_clean, det_clean["odd_count"] / tot_clean]
    stego_ratios = [det_stego["even_count"] / tot_stego, det_stego["odd_count"] / tot_stego]

    x_pos = np.arange(2)
    w_bar = 0.35
    ax_bars.bar(x_pos - w_bar/2, [clean_ratios[0], stego_ratios[0]], width=w_bar, label="Partall (h%2==0)", color="#2ca02c")
    ax_bars.bar(x_pos + w_bar/2, [clean_ratios[1], stego_ratios[1]], width=w_bar, label="Oddetall (h%2!=0)", color="#ff7f0e")
    ax_bars.axhline(0.5, color='black', linestyle='--', linewidth=1, label="50/50 likevekt")
    ax_bars.set_xticks(x_pos)
    ax_bars.set_xticklabels(cat_labels, fontsize=9)
    ax_bars.set_title("Paritet i nære diff (|h| ≤ 10)", fontsize=10, fontweight='bold')
    ax_bars.set_ylabel("Andel av nabopar", fontsize=8)
    ax_bars.set_ylim(0, 0.9)
    ax_bars.legend(fontsize=7, loc="upper right")
    ax_bars.grid(True, linestyle=":", alpha=0.5)

    # Panel [2, 3]: Imbalance-metrikk direkte mot terskelverdien
    ax_metric = axes[2, 3]
    metrics = [det_clean["imbalance"], det_stego["imbalance"]]
    bars = ax_metric.bar(["Original", "Stego"], metrics, color=["#1f77b4", "#d62728"], width=0.5)
    ax_metric.axhline(0.035, color="red", linestyle=":", lw=1.5, label="Terskel (0.035)")
    ax_metric.set_title("Deteksjonsmetrikk (Imbalance)", fontsize=10, fontweight='bold')
    ax_metric.set_ylabel("(Partall - Oddetall) / Totalt", fontsize=8)
    ax_metric.legend(fontsize=8)
    ax_metric.grid(True, linestyle=":", alpha=0.5)
    for b in bars:
        h = b.get_height()
        ax_metric.annotate(f"{h:.4f}", xy=(b.get_x() + b.get_width() / 2, h),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    # Panel [2, 4]: Tekstoppsummering av konklusjonsgrunnlaget
    ax_text = axes[2, 4]
    ax_text.axis('off')
    stat_summary = (
        "STATISTISK GRUNNLAG FOR KONKLUSJON:\n\n"
        f"1. Original imbalance: {det_clean['imbalance']:.4f}\n"
        "   - Naturlig bilde har skarp Laplace-topp\n"
        "   - h=0 (partall) dominerer kraftig.\n\n"
        f"2. Stego imbalance: {det_stego['imbalance']:.4f}\n"
        "   - Ved DE: h' = 2h + b\n"
        "   - b innsettes tilfeldig (50% 0, 50% 1)\n"
        "   - Skaper kunstig likevekt mellom\n"
        "     partall og oddetall.\n\n"
        f"3. Terskel: 0.0350\n"
        f"   -> Resultat: {det_stego['konklusjon']}"
    )
    ax_text.text(0.05, 0.5, stat_summary, fontsize=9, verticalalignment='center',
                 family='monospace', bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="#ccc"))

    plt.savefig(ANALYSIS_OUTPUT, dpi=200)
    print(f"Oversiktsfigur lagret til: '{ANALYSIS_OUTPUT}'")
    plt.show()

if __name__ == "__main__":
    main()
