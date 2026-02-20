<div align="center">

![EPOQ](image-trainer/public/epoq3.png)

**A powerful, cross-platform desktop application for training image classification models.**

[![Tauri Version](https://img.shields.io/badge/Tauri-v2-blue?logo=tauri)](https://tauri.app)
[![Next.js](https://img.shields.io/badge/Next.js-v14-black?logo=next.js)](https://nextjs.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-v2.0+-red?logo=pytorch)](https://pytorch.org)
[![Rust](https://img.shields.io/badge/Rust-v1.75+-orange?logo=rust)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Features](#-features) • [Prerequisites](#-prerequisites) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Contributing](#-contributing)

</div>

---

## 📖 Overview

**EPOQ** brings the power of deep learning to your desktop. Built with **Tauri**, **Next.js**, and **PyTorch**, this application provides a sleek, intuitive GUI for training state-of-the-art image classification models locally. Say goodbye to complex CLI scripts and hello to real-time visual feedback!

## ✨ Features

- 🗂️ **Smart Dataset Management**: Easily select folders with `train`/`val` structures, or let the app auto-split flat folders. Export your prepared dataset to a ZIP file with one click.
- ⚙️ **Flexible Model Configuration**: Choose from modern architectures including **ResNet18**, **DCN**, and **EVA02**. Customize training epochs and batch sizes to suit your hardware.
- 📈 **Real-Time Analytics**: Monitor the training process with live PyTorch logs and dynamic, interactive **Loss and Accuracy Charts** (powered by Recharts).
- 📊 **Comprehensive Results**: After training completes, dive into detailed metrics with a full **Classification Report** and a visual **Confusion Matrix**.
- 🚀 **Native Performance**: Engineered with Rust and Tauri for a lightweight desktop footprint, paired with a blazing-fast Next.js frontend.

## 🛠️ Tech Stack

- **Frontend**: Next.js, React, Tailwind CSS, Framer Motion, Lucide Icons, Recharts
- **Backend/Desktop Platform**: Tauri, Rust
- **Machine Learning**: Python, PyTorch, Torchvision, Scikit-learn, Pandas, Matplotlib, Seaborn

---

## 🚀 Prerequisites

Before you begin, ensure you have the following installed on your system:

1. **[Node.js](https://nodejs.org/)** (v18+)
2. **[Rust](https://rustup.rs/)** (Required for Tauri builds)
3. **[Python](https://www.python.org/downloads/)** (3.9+)
4. **Machine Learning Dependencies**:
   ```bash
   pip install torch torchvision pandas scikit-learn matplotlib seaborn
   ```

---

## 💻 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Sree14hari/EPOQ.git
   cd EPOQ
   ```

2. **Install Node dependencies**

   ```bash
   cd image-trainer
   npm install
   ```

3. **Start Development Server**
   ```bash
   npm run tauri dev
   ```
   _This command will compile the Rust backend, start the Next.js frontend, and launch the desktop window._

---

## 📦 Building for Production

To create an optimized, standalone executable for your operating system:

```bash
npm run tauri build
```

Once the build is complete, you can find the executable in `src-tauri/target/release/`.

---

## 🧩 Architecture

The application marries web technologies with native system capabilities:

1. **Next.js Frontend**: Provides a highly responsive, modern UI styled with Tailwind CSS.
2. **Tauri Core**: Written in Rust, it acts as the bridge, managing system menus, dialogs, and native operations.
3. **Python Worker Runtime**: Tauri spawns a Python subprocess (`python_backend/script.py`) to execute PyTorch training scripts securely and stream the stdout/stderr back to the UI for live tracking.

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">
  Made with ❤️ by the Sreehari R
</div>
