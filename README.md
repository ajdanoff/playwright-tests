# Image Crawler and Classification Pipeline

A Python-based asynchronous web crawler that scrapes images from search engines (especially Bing), downloads images with metadata, and trains deep learning classification models on the crawled data using PyTorch.

---

## Features

- Asynchronous image crawling and downloading using Playwright and asyncio.
- Robots.txt compliance and respectful crawl-delay enforcement.
- Modular search service abstraction supporting Bing and Google.
- Flexible image metadata annotation with CSV export.
- Deep learning model pipeline with PyTorch for training image classifiers.
- Automated tests with pytest for crawler and model components.
- Cross-browser end-to-end testing with Playwright.
- Continuous Integration using GitHub Actions running both pytest and Playwright tests.

---

## Project Structure

```
├── crawler/ # Web crawler implementation modules
│ ├── images_crawler.py # ImagesCrawler class and logic
│ ├── search_services.py # SearchService interface and implementations
│ ├── html_parser.py # Custom HTML image tags parser
│ └── ...
├── ml/ # Machine learning model training pipeline
│ ├── train_model.py # MLModel class using PyTorch
│ └── tests/ # Unit tests for ML and crawler modules
├── data/ # Folder for downloaded images and annotations
├── requirements.txt # Required Python packages
└── README.md # This file
```
---

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js and npm (for Playwright)
- Playwright installed with browsers:


```commandline
python -m pip install playwright
playwright install
```
- Other dependencies listed in `requirements.txt`. Install using:
`python -m pip install -r requirements.txt`


---

## Usage

### Running the Image Crawler

1. Set target search query and data folder in `ImagesCrawler`.
2. Run asynchronous crawl routine to scrape and download images.
3. Annotations (image filename, label, dimensions) are saved as CSV.

### Training the ML Model

1. Configure `MLModel` with dataset folders and annotation CSVs.
2. Define hyperparameters like epochs, batch size, learning rate.
3. Call `train_epochs()` to train and evaluate the model.
4. Use provided unit tests to validate the pipeline.

---

## Testing

### Running pytest Tests

Run all Python unit and integration tests located in the `tests/` directory:

```commandline
pytest tests --maxfail=1 --disable-warnings -q
```


Tests cover:

- Image crawling and downloading verification.
- ML model input/output shape correctness.
- Training step execution and multi-epoch runs.

### Running Playwright Tests

Run Playwright end-to-end browser tests in headed (visible) mode:

`npx playwright test tests --headed`

---


Playwright tests cover cross-browser compatibility on Chromium, Firefox, and WebKit.

---

## Continuous Integration (CI)

The project uses GitHub Actions to run tests automatically on push and pull requests to `main` and `master` branches.

### Features of CI Workflow

- Checks out the code repository.
- Sets up Python 3.11 and Node.js for Playwright.
- Installs dependencies and Playwright browsers.
- Starts a virtual display server (Xvfb) for headed testing.
- Runs pytest tests in `tests/`.
- Runs Playwright tests in headed mode in `tests/`.
- Uploads Playwright traces and test results for debugging.

### GitHub Actions Workflow Example (`.github/workflows/playwright.yml`)

```yaml
name: Playwright and Pytest Tests

on:
  push:
    branches: [main, master]
  pull_request:
    branches: [main, master]

jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 60

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Install Playwright browsers
        run: python -m playwright install --with-deps

      - name: Install Xvfb
        run: sudo apt-get update && sudo apt-get install -y xvfb

      - name: Start Xvfb
        run: |
          Xvfb :99 -screen 0 1280x720x24 &
          echo "DISPLAY=:99" >> $GITHUB_ENV

      - name: Run pytest tests in 'tests' directory
        env:
          DISPLAY: :99
        run: |
          pytest tests --maxfail=1 --disable-warnings -q

      - name: Run Playwright tests in headed mode from 'tests' subfolder
        env:
          DISPLAY: :99
        run: |
          npx playwright test tests --headed

      - name: Upload Playwright traces
        uses: actions/upload-artifact@v4
        if: ${{ !cancelled() }}
        with:
          name: playwright-traces
          path: test-results/
```

---

## Architecture

The pipeline is composed of these primary components:

- **Crawler module**: Asynchronously fetches and downloads images from search engines, enforcing robots.txt and respecting crawl delays.
- **Metadata annotator**: Extracts and organizes image metadata into CSV files for supervised training.
- **Machine Learning module**: Implements image classification using PyTorch with configurable hyperparameters, supporting training and testing modes.
- **Testing suite**: Unit and integration tests with pytest assure code correctness, combined with Playwright tests for browser UI level validation.
- **CI/CD pipeline**: Automated testing via GitHub Actions integrates both Python tests and UI tests, supporting headed mode for browser visibility during debugging.

---

## Roadmap

Planned future enhancements include:

- Implementing advanced CNN architectures such as ResNet or EfficientNet for improved accuracy.
- Incorporating transfer learning techniques using pretrained models.
- Enhancing crawler capabilities to include object detection and localization.
- Deploying trained models as scalable web services with user-friendly frontends.
- Adding support for active learning to reduce labeling effort.
- Extending coverage of cross-browser Playwright tests to mobile emulation.
- Improving test coverage and adding performance benchmarking.

---

## FAQ

**Q: What search engines are supported for crawling?**  
A: Currently, the pipeline supports Bing and Google via modular search services.

**Q: How is the crawler respectful to website policies?**  
A: It parses and obeys robots.txt files and respects crawl-delay directives to avoid overloading servers.

**Q: Can I customize the ML model architecture?**  
A: Yes, the MLModel class is modular and customizable with different network architectures and training parameters.

**Q: How do I add new tests?**  
A: Add pytest tests to the `tests/` directory for backend logic and Playwright tests also in `tests/` for UI flows.

**Q: How can I debug Playwright tests locally?**  
A: Run `npx playwright test tests --headed` to see browser actions visually and use Playwright tracing tools.

---

## Contributing

Contributions, bug reports, and feature requests are welcome!  
Feel free to fork the repository, make changes, and submit pull requests.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

Alexander Zhdanov – [alexander.jdanoff@gmail.com](mailto:alexander.jdanoff@gmail.com)  
Project repository: [https://github.com/ajdanoff/playwright-tests](https://github.com/ajdanoff/playwright-tests)

