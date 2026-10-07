import os
import struct
import zlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import spline_filter, map_coordinates, median_filter
from scipy.fftpack import dct
from scipy.stats import chi2

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

SECRET_KEY = 42819
SPLINE_ORDER = 3
TARGET_SECRET_SIZE = (128, 128)

# =====================================================================
# 1. B-SPLINE MODELLERING & KRYPTERING
# =====================================================================

def get_bspline_coeffs(image: np.ndarray, order: int = 3) -> np.ndarray:
    return spline_filter(image.astype(np.float32), order=order)

def evaluate_bspline(coeffs: np.ndarray, shape: tuple[int, int], order: int = 3) -> np.ndarray:
    coords = np.indices(shape, dtype=np.float32)
    reconstructed = map_coordinates(coeffs, coords, order=order, mode='mirror', prefilter=False)
    return np.clip(np.round(reconstructed), 0, 255).astype(np.uint8)

def encrypt_coeffs(coeffs: np.ndarray, key: int) -> bytes:
    h, w = coeffs.shape
    c_min, c_max = float(coeffs.min()), float(coeffs.max())
    
    norm = (coeffs - c_min) / (c_max - c_min + 1e-8)
    q_coeffs = np.clip(np.round(norm * 255), 0, 255).astype(np.uint8)
    
    rng = np.random.RandomState(key)
    flat = q_coeffs.flatten()
    n = len(flat)
    
    perm = rng.permutation(n)
    shuffled = flat[perm]
    keystream = rng.randint(0, 256, size=n, dtype=np.uint8)
    encrypted = np.bitwise_xor(shuffled, keystream)
    
    header = struct.pack(">HHffI", h, w, c_min, c_max, n)
    return header + encrypted.tobytes()

def decrypt_coeffs(payload: bytes, key: int) -> tuple[np.ndarray, tuple[int, int]]:
    header = payload[:16]
    h, w, c_min, c_max, n = struct.unpack(">HHffI", header)
    enc_data = np.frombuffer(payload[16:16+n], dtype=np.uint8)
    
    rng = np.random.RandomState(key)
    perm = rng.permutation(n)
    keystream = rng.randint(0, 256, size=n, dtype=np.uint8)
    
    unmasked = np.bitwise_xor(enc_data, keystream)
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(n)
    q_coeffs = unmasked[inv_perm]
    
    dequant = q_coeffs.astype(np.float32) / 255.0
    restored = dequant * (c_max - c_min) + c_min
    return restored.reshape((h, w)), (h, w)

# =====================================================================
# 2. REVERSIBEL LSB INNBAKING & GJENOPPRETTING
# =====================================================================

def embed_reversible_lsb(cover: np.ndarray, secret_payload: bytes) -> tuple[np.ndarray, int]:
    flat = cover.copy().flatten()
    secret_len = len(secret_payload)
    raw_secret_bits = np.unpackbits(np.frombuffer(secret_payload, dtype=np.uint8))
    
    orig_lsbs = flat[:len(raw_secret_bits)] & 1
    comp_orig = zlib.compress(np.packbits(orig_lsbs).tobytes(), level=9)
    
    total_est = 12 + secret_len + len(comp_orig)
    orig_lsbs_all = flat[:total_est * 8] & 1
    comp_orig_all = zlib.compress(np.packbits(orig_lsbs_all).tobytes(), level=9)
    
    total_mod_bits = (12 + secret_len + len(comp_orig_all)) * 8
    final_header = struct.pack(">III", secret_len, len(comp_orig_all), total_mod_bits)
    final_payload = final_header + secret_payload + comp_orig_all
    final_bits = np.unpackbits(np.frombuffer(final_payload, dtype=np.uint8))
    
    if len(final_bits) > len(flat):
        raise ValueError("Cover er for lite for payload + komprimerte reverseringsdata.")
        
    flat[:len(final_bits)] = (flat[:len(final_bits)] & 0xFE) | final_bits
    return flat.reshape(cover.shape), len(final_bits)

def extract_and_restore(stego: np.ndarray) -> tuple[bytes, np.ndarray]:
    flat = stego.copy().flatten()
    header_bytes = np.packbits(flat[:96] & 1).tobytes()
    secret_len, comp_len, _ = struct.unpack(">III", header_bytes)
    
    total_bytes = 12 + secret_len + comp_len
    raw_payload = np.packbits(flat[:total_bytes * 8] & 1).tobytes()
    
    secret_payload = raw_payload[12 : 12 + secret_len]
    comp_orig_all = raw_payload[12 + secret_len : 12 + secret_len + comp_len]
    
    decomp_orig = zlib.decompress(comp_orig_all)
    original_bits = np.unpackbits(np.frombuffer(decomp_orig, dtype=np.uint8))[:total_bytes * 8]
    
    restored_flat = flat.copy()
    restored_flat[:len(original_bits)] = (restored_flat[:len(original_bits)] & 0xFE) | original_bits
    return secret_payload, restored_flat.reshape(stego.shape)

# =====================================================================
# 3. SIGNAL- OG STATISTISKE ANALYSEFUNKSJONER
# =====================================================================

def compute_fft_spectrum(img: np.ndarray) -> np.ndarray:
    """Beregner sentrert logaritmisk 2D Fourier-spektrum."""
    f = np.fft.fft2(img.astype(np.float64))
    fshift = np.fft.fftshift(f)
    return np.log1p(np.abs(fshift))

def compute_block_dct_energy(img: np.ndarray, block_size: int = 8) -> np.ndarray:
    """Beregner akkumulert høyfrekvens-energi for 8x8 DCT-blokker."""
    h, w = img.shape
    h_blocks = h // block_size
    w_blocks = w // block_size
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
            blk = img[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size].astype(np.float64)
            coeff = dct2(blk)
            energy_map[i, j] = np.sum(np.abs(coeff[hf_mask]))
            
    return energy_map

def compute_noise_residual(img: np.ndarray) -> np.ndarray:
    """Beregner støyresidual ved differanse mot et 3x3 medianfilter."""
    filtered = median_filter(img, size=3)
    residual = np.abs(img.astype(np.int16) - filtered.astype(np.int16))
    return np.clip(residual * 8, 0, 255).astype(np.uint8)

def compute_chi_square(image: np.ndarray, block_size: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    flat = image.flatten()
    n = len(flat)
    x_pct, probs = [], []
    counts = np.zeros(256, dtype=np.int64)
    
    for i in range(0, n, block_size):
        blk = flat[i:i + block_size]
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
        
        df = k - 1
        x_pct.append(((i + len(blk)) / n) * 100.0)
        probs.append(1.0 - (1.0 - chi2.cdf(chi_stat, df)))
        
    return np.array(x_pct), np.array(probs)

# =====================================================================
# 4. HOVEDPROGRAM OG VISUALISERING
# =====================================================================

def main():
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR, exist_ok=True)
        
    # Syntetisk fallback dersom filer mangler
    if not os.path.exists(COVER_PATH):
        print(f"Genererer test-cover: {COVER_PATH}")
        y, x = np.mgrid[0:720, 0:1280]
        grad = np.clip((np.sin(x/35.0)*45 + np.cos(y/35.0)*45 + 128) + np.random.normal(0, 3, (720, 1280)), 0, 255)
        Image.fromarray(grad.astype(np.uint8)).save(COVER_PATH)
        
    if not os.path.exists(SECRET_PATH):
        print(f"Genererer hemmelig testbilde: {SECRET_PATH}")
        sec = np.zeros((128, 128), dtype=np.uint8)
        sec[32:96, 32:96] = 210
        Image.fromarray(sec).save(SECRET_PATH)

    print(f"Laster inn kildebilder fra {BASE_DIR}...")
    cover_img = np.array(Image.open(COVER_PATH).convert('L'), dtype=np.uint8)
    secret_raw = Image.open(SECRET_PATH).convert('L').resize(TARGET_SECRET_SIZE, Image.Resampling.LANCZOS)
    secret_img = np.array(secret_raw, dtype=np.uint8)
    
    # 1. B-spline kryptering og innbaking
    coeffs = get_bspline_coeffs(secret_img, order=SPLINE_ORDER)
    secret_payload = encrypt_coeffs(coeffs, key=SECRET_KEY)
    stego_img, total_bits_mod = embed_reversible_lsb(cover_img, secret_payload)
    Image.fromarray(stego_img).save(STEGO_OUTPUT)
    cap_pct = (total_bits_mod / cover_img.size) * 100.0
    
    # 2. Uthenting og full rekonstruksjon
    extracted_secret, restored_cover_img = extract_and_restore(stego_img)
    dec_coeffs, secret_shape = decrypt_coeffs(extracted_secret, key=SECRET_KEY)
    restored_secret = evaluate_bspline(dec_coeffs, shape=secret_shape, order=SPLINE_ORDER)
    
    Image.fromarray(restored_secret).save(RESTORED_SECRET_OUTPUT)
    Image.fromarray(restored_cover_img).save(RESTORED_COVER_OUTPUT)
    is_exact = np.array_equal(cover_img, restored_cover_img)
    print(f"Dekryptering fullfort. Originalt cover gjenopprettet: {is_exact}")

    # 3. Beregn analysedata
    print("Beregner Fourier-spekter, DCT-fordeling og stoyavvik...")
    lsb_clean = (cover_img & 1) * 255
    lsb_stego = (stego_img & 1) * 255
    fft_clean = compute_fft_spectrum(cover_img)
    fft_stego = compute_fft_spectrum(stego_img)
    dct_clean = compute_block_dct_energy(cover_img, block_size=8)
    dct_stego = compute_block_dct_energy(stego_img, block_size=8)
    res_clean = compute_noise_residual(cover_img)
    res_stego = compute_noise_residual(stego_img)
    x_c, p_c = compute_chi_square(cover_img)
    x_s, p_s = compute_chi_square(stego_img)

    # 4. Ryddig visualisering (constrained_layout forhindrer overlapp)
    fig, axes = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)
    fig.suptitle(
        f"Sammenlignende Signal- og Frekvensanalyse (Modifisert omrade: {cap_pct:.1f} %)",
        fontsize=15, fontweight='bold'
    )

    # --- RAD 1: ROM-DOMENE & LSB ---
    axes[0, 0].imshow(cover_img, cmap='gray')
    axes[0, 0].set_title("Originalt bilde", fontsize=11, pad=6)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(stego_img, cmap='gray')
    axes[0, 1].set_title("Bilde med innvevd data", fontsize=11, pad=6)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(lsb_clean, cmap='gray')
    axes[0, 2].set_title("Originalt LSB-plan", fontsize=11, pad=6)
    axes[0, 2].axis('off')

    axes[0, 3].imshow(lsb_stego, cmap='gray')
    axes[0, 3].set_title("Modifisert LSB-plan", fontsize=11, pad=6)
    axes[0, 3].axis('off')

    # --- RAD 2: FREKVENSDOMENE (FOURIER & DCT) ---
    axes[1, 0].imshow(fft_clean, cmap='viridis')
    axes[1, 0].set_title("Fourier-spektrum (Original)", fontsize=11, pad=6)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(fft_stego, cmap='viridis')
    axes[1, 1].set_title("Fourier-spektrum (Modifisert)", fontsize=11, pad=6)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(dct_clean, cmap='magma')
    axes[1, 2].set_title("8x8 DCT HF-energi (Original)", fontsize=11, pad=6)
    axes[1, 2].axis('off')

    axes[1, 3].imshow(dct_stego, cmap='magma')
    axes[1, 3].set_title("8x8 DCT HF-energi (Modifisert)", fontsize=11, pad=6)
    axes[1, 3].axis('off')

    # --- RAD 3: STØYRESIDUALER, STATISTIKK OG REKONSTRUKSJON ---
    axes[2, 0].imshow(res_clean, cmap='gray')
    axes[2, 0].set_title("Stoyresidual (Original)", fontsize=11, pad=6)
    axes[2, 0].axis('off')

    axes[2, 1].imshow(res_stego, cmap='gray')
    axes[2, 1].set_title("Stoyresidual (Modifisert)", fontsize=11, pad=6)
    axes[2, 1].axis('off')

    # Chi-kvadrat plott
    ax_plot = axes[2, 2]
    ax_plot.plot(x_c, p_c, label="Original", color="#1f77b4", lw=1.5)
    ax_plot.plot(x_s, p_s, label="Modifisert", color="#d62728", lw=1.8)
    ax_plot.axvline(x=cap_pct, color="black", linestyle="--", lw=1.2, label=f"Slutt ({cap_pct:.1f} %)")
    ax_plot.set_title(r"Chi-kvadrat ($\chi^2$) fordeling", fontsize=11, pad=6)
    ax_plot.set_xlabel("Bildeflate inspisert (%)", fontsize=9)
    ax_plot.set_ylabel(r"Sannsynlighet ($1 - p$)", fontsize=9)
    ax_plot.set_ylim(-0.05, 1.05)
    ax_plot.tick_params(labelsize=8)
    ax_plot.grid(True, linestyle=":", alpha=0.6)
    ax_plot.legend(loc="center right", fontsize=8)

    # Rekonstruert hemmelig bilde
    axes[2, 3].imshow(restored_secret, cmap='gray')
    axes[2, 3].set_title("Rekonstruert B-spline", fontsize=11, pad=6)
    axes[2, 3].axis('off')

    # Lagre høyoppløselig figur
    plt.savefig(ANALYSIS_OUTPUT, dpi=200)
    print(f"Oversiktsfigur lagret til: '{ANALYSIS_OUTPUT}'")
    plt.show()

if __name__ == "__main__":
    main()
