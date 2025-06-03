# Eidocell 🔬

Eidocell is a desktop application designed for interactive image analysis. It provides a suite of tools for exploring image datasets, performing segmentation, classification, clustering, and visualizing results.

> **Note:** Eidocell is currently in an early stage of development. Features may be incomplete, and the application might be unstable. We appreciate your understanding and feedback!

## Features (Current & Planned)

*   **Interactive Image Gallery:** Browse and manage your image datasets.
*   **Segmentation Tools:**
    *   Base segmentation methods (Otsu, Watershed, Adaptive Thresholding).
    *   Advanced, model-driven segmentation capabilities (under development).
*   **Image Classification & Clustering:** Group and categorize images based on their features.
*   **Data Analysis & Plotting:** Generate histograms and scatter plots to analyze image-derived data.
*   **Session Management:** Save and load your analysis sessions.

## Installation

### Using `venv` (Recommended for pip-based projects)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/eidocell.git
    cd eidocell
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    # venv\Scripts\activate
    # On macOS/Linux
    # source venv/bin/activate
    ```

3.  **Install dependencies:**
    It is highly recommended to use a `requirements.txt` file. If one is provided:
    ```bash
    pip install -r requirements.txt
    ```

### Using Conda

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/eidocell.git
    cd eidocell
    ```

2.  **Create and activate a Conda environment:**
    ```bash
    conda create -n eidocell-env python=3.9
    conda activate eidocell-env
    ```

3.  **Install dependencies:**
    If you created the environment manually, install dependencies. You can use `pip` within a Conda environment:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Adding model weights:**
    After successful cloning and setup of the environment, you need to download model weights and put them in the
    ```bash
    src/backend/resources
    ```
    Download the archive [from here](https://drive.google.com/file/d/1n5KeIvGgWTyxuStDs3n2bw67nPBmHDsP/view).

## Running Eidocell

Once dependencies are installed, run the main application window from your activated environment:
```bash
python src/UI/main_window.py
