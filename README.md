# DateGrade
**Assigning a grade to date fruits based on external quality features using EfficientNetB3.**

> **Live Demo:** Access the integrated Gradio interface [here](https://huggingface.co/spaces/Korj14/date-fruit-quality-grading).

---

## Dataset
* **Source:** Mendeley Data - [Date Fruit Dataset](https://data.mendeley.com/datasets/s5zfvsw5kv/3)
* **Scope:** Used the full **Gajar** directory (all grades and sizes) from the original 'Date Fruit' directory (not the pre-augmented set).
* **Pre-processing:** Rearranged all sizes into consolidated folders for **Grade 1, Grade 2, and Grade 3**.
* **Colab Tip:** To download directly to Colab/Drive, right-click the 'Download All' button on Mendeley to get the direct link for your `wget` or `curl` commands.

## Tools & Hardware
* **Environment:** Developed on **Google Colab**.
* **Hardware:** Recommended use of free hardware accelerators (**GPU** or **TPU**).

## Explanation
* **Detailed Report:** For references, related work, and methodology, see the [Comprehensive_Report](Comprehensive_Report.docx) file in the project folder.
* **Architecture:** For a concise plan and network breakdown, see [Training_and_Deployment_Plan_and_Architecture](Training_and_Deployment_Plan_and_Architecture.docx) in the project folder.

## Files
* `dataset.py`: Manages data loading, preprocessing, and the online augmentation pipeline.
* `model.py`: Contains the CNN architecture definition (EfficientNetB3) and specialized FGVC modules.
* `utils.py`: Helper functions for logging, metric calculations, and image processing.
* `config.yaml`: Centralized configuration for hyperparameters (learning rate, batch size, etc.) and file paths.
* `main_training.ipynb`: Primary entry point; handles training, evaluation, and monitoring.
* `logs/`: Stores training history, loss curves, and evaluation results.
* `exports/`: Contains final serialized models ready for deployment or inference.
* `checkpoints/`: Stores saved PyTorch model weights (`.pth`) during training for later recovery.

## How to use
> **Note:** This setup is optimized for small datasets but remains adaptable. Some design choices were experimental and may require customization for other tasks.

1. **Setup:** Import the dataset into the project directory.
2. **Clean:** Clear the contents of `logs/`, `exports/`, and `checkpoints/` folders.
3. **Execution:** Open `main_training.ipynb` and run numbered cells in sequence.
4. **Resuming:** Only run **'Cell 6 (if resuming from checkpoint)'** if the primary training cell is interrupted or cut off.
   * *Note: Cell 9 is optional/not required for standard training.*

> **Note:** The All three folders still contain results from our training run. You may delete their contents entirely if it causes confusion when rerunning.
