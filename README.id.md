# torch-opencl

Ekstensi Python yang mengintegrasikan PyTorch dengan OpenCL, dibangun menggunakan [nanobind](https://github.com/wjakob/nanobind) dan [scikit-build-core](https://github.com/scikit-build/scikit-build-core).

## Persyaratan Sistem

- Python >= 3.10
- CMake >= 3.15
- Kompiler C++ dengan dukungan C++17
- [uv](https://github.com/astral-sh/uv) (manajer paket yang direkomendasikan)

## Dependensi

| Paket | Versi |
|-------|-------|
| `torch` | >= 2.11 |
| `numpy` | < 2.4 |
| `nanobind` | >= 2.12.0 |
| `scikit-build-core` | >= 0.12 |

> **Catatan:** Pada Linux, PyTorch akan diunduh dari indeks PyTorch CPU (`https://download.pytorch.org/whl/cpu`).

## Instalasi

Instalasi **wajib menggunakan uv** karena pengelolaan dependensi dan proses build diatur melalui konfigurasi `uv` di `pyproject.toml`.

```bash
# Clone repositori
git clone <url-repositori>
cd torch-opencl

# Instal dependensi dan build ekstensi
uv sync

# Aktifkan environment virtual
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### Dukungan LSP (Language Server Protocol)

Karena `uv sync` menjalankan build secara terisolasi, direktori `build/` beserta artefak sumber akan dihapus setelah proses selesai. Akibatnya, LSP seperti `clangd` (untuk C++) maupun `pylsp`/`pyright` (untuk Python) tidak dapat menemukan file sumber yang dibutuhkan.

Untuk mengatasi hal ini, jalankan perintah berikut setelah `uv sync`:

```bash
uv pip install . --no-build-isolation
```

Perintah ini membangun ulang ekstensi **tanpa isolasi**, sehingga direktori `build/` tetap tersimpan dan LSP dapat mengenali sumber C++ maupun Python dengan benar.

## Struktur Proyek

```
torch-opencl/
├── src/
│   ├── main.cpp                  # Implementasi ekstensi C++
│   └── torch_opencl/
│       └── __init__.py           # Entry point modul Python
├── tests/
│   └── test_main.py              # Unit test
├── CMakeLists.txt                # Konfigurasi build CMake
├── pyproject.toml                # Konfigurasi proyek Python
└── uv.lock                       # Lock file dependensi
```

## Penggunaan

```python
import torch_opencl

# Contoh: fungsi penjumlahan
hasil = torch_opencl.add(1, 2)
print(hasil)  # Output: 3
```

## Build

```bash
uv build
```

## Pengujian

```bash
# Tanpa mengaktifkan virtual environment
uv run pytest

# Jika virtual environment sudah aktif
pytest
```

## Lisensi

Lihat file `LICENSE` untuk informasi lisensi lebih lanjut.
