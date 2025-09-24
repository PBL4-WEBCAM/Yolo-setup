# YOLO Object Detection Web Application

A Django web application for real-time object detection using YOLOv8 models with webcam integration.

## Features

- Real-time object detection using YOLOv8
- Webcam integration for live detection
- Web-based interface
- Support for multiple YOLO model variants

## Prerequisites

- Python 3.8 or higher
- Git
- Git LFS (for large model files)
- Webcam/camera device

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/PBL4-WEBCAM/Yolo-setup.git
cd Yolo-setup
```

### 2. Install Git LFS (if not already installed)

**On macOS:**
```bash
brew install git-lfs
```

**On Ubuntu/Debian:**
```bash
sudo apt-get install git-lfs
```

**On Windows:**
Download from [Git LFS website](https://git-lfs.github.com/)

### 3. Initialize Git LFS and Pull Model Files

```bash
git lfs install
git lfs pull
```

### 4. Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist, install the main dependencies:

```bash
pip install django
pip install ultralytics
pip install opencv-python
pip install pillow
pip install torch torchvision
```

### 6. Database Migration

```bash
python manage.py migrate
```

### 7. Run the Application

```bash
python manage.py runserver
```

The application will be available at `http://127.0.0.1:8000/`

## Usage

1. Open your web browser and navigate to `http://127.0.0.1:8000/`
2. Allow camera permissions when prompted
3. Click on the detection interface
4. The application will start detecting objects in real-time from your webcam

## Model Information

This application uses YOLOv8 models for object detection. The model files are stored using Git LFS due to their large size (>100MB).

### Available Models

- `yolov8x-oiv7.pt` - YOLOv8 Extra Large model trained on Open Images V7

## Project Structure

```
cam/
├── detection/              # Django app for object detection
│   ├── templates/         # HTML templates
│   ├── views.py          # View functions
│   └── __pycache__/      # Python cache files
├── yolov8x-oiv7.pt       # YOLO model file (Git LFS)
├── manage.py             # Django management script
└── .gitattributes        # Git LFS configuration
```

## Troubleshooting

### Git LFS Issues

If you encounter issues with large files:

```bash
# Reinstall Git LFS hooks
git lfs install --force

# Re-download LFS files
git lfs fetch
git lfs checkout
```

### Camera Permission Issues

- Ensure your browser has camera permissions
- Check if another application is using the camera
- Try refreshing the browser page

### Model Loading Issues

If the YOLO model fails to load:

1. Verify the model file exists: `ls -la yolov8x-oiv7.pt`
2. Check if Git LFS downloaded the file properly
3. Re-run `git lfs pull` to ensure model files are downloaded

### Python Dependencies

If you encounter import errors:

```bash
# Upgrade pip
pip install --upgrade pip

# Reinstall packages
pip install --force-reinstall -r requirements.txt
```

## Development

### Adding New Models

1. Add the model file to the repository
2. Configure Git LFS tracking:
   ```bash
   git lfs track "*.pt"
   git add .gitattributes
   ```
3. Add and commit the model file
4. Update the view functions to use the new model

### Creating requirements.txt

If `requirements.txt` doesn't exist, create it:

```bash
pip freeze > requirements.txt
```

## Requirements

Core dependencies typically include:

```
Django>=4.0.0
ultralytics
opencv-python
Pillow
torch
torchvision
```

### Environment Variables

Create a `.env` file for environment-specific settings:

```
DEBUG=True
SECRET_KEY=your-secret-key
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

[Add your license information here]

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify all prerequisites are installed
3. Ensure your camera is working properly
4. Check the browser console for JavaScript errors

For technical support, please create an issue on the GitHub repository.
