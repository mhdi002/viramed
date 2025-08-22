# Medical AI Backend System - API Documentation

## Overview

The Medical AI Backend System is a comprehensive platform for managing and running inference on various medical AI models. It supports multiple AI frameworks (YOLO, TensorFlow, PyTorch) and provides specialized services for different medical domains.

## Architecture

The system follows a Model-View-Controller (MVC) architecture:

- **Models**: Data structures and database schemas
- **Views**: API response formatting and validation (Pydantic models)
- **Controllers**: Request handling and business logic
- **Services**: AI model inference services

## Authentication

The API uses JWT (JSON Web Token) authentication. Include the token in the Authorization header:

```
Authorization: Bearer <your_jwt_token>
```

### User Roles

- **Admin**: Full system access, user management
- **Doctor**: Inference access, view models and reports
- **Researcher**: Inference access, batch processing, data export
- **Viewer**: View-only access to models

## Base URL

```
http://localhost:8001
```

All API endpoints are prefixed with `/api`.

## Endpoints

### Authentication Endpoints

#### POST /api/auth/register
Register a new user (Admin only)

**Request Body:**
```json
{
  "username": "string",
  "email": "user@example.com",
  "password": "string",
  "full_name": "string",
  "role": "viewer|doctor|researcher|admin"
}
```

#### POST /api/auth/login
User login

**Request Body:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "access_token": "string",
  "token_type": "bearer",
  "expires_in": 1800,
  "user": {
    "id": "string",
    "username": "string",
    "email": "user@example.com",
    "role": "string",
    "permissions": ["string"]
  }
}
```

#### GET /api/auth/me
Get current user information

### Model Management Endpoints

#### GET /api/models/
List all medical models

**Query Parameters:**
- `domain`: Filter by medical domain (optional)
- `model_type`: Filter by model type (optional)

**Response:**
```json
[
  {
    "id": "string",
    "name": "string",
    "filename": "string",
    "model_type": "yolo|tensorflow|pytorch",
    "task_type": "detection|classification|segmentation|regression",
    "medical_domain": "mammography|bone_age|retina|liver|colon|brain|ms_segmentation",
    "format": "pt|pth|h5|onnx",
    "description": "string",
    "classes": ["string"],
    "is_active": true,
    "created_at": "2025-01-01T00:00:00Z"
  }
]
```

#### GET /api/models/{model_name}
Get specific model by name

#### POST /api/models/
Create new model entry

#### POST /api/models/upload
Upload and register a new model file

**Form Data:**
- `file`: Model file (.pt, .pth, .h5, .onnx)
- `name`: Model name
- `model_type`: Model framework type
- `task_type`: AI task type
- `medical_domain`: Medical domain
- `description`: Model description (optional)
- `input_size`: Input image size (optional)
- `classes`: JSON string of class names (optional)

#### POST /api/models/refresh
Refresh models from storage directory

#### PUT /api/models/{model_name}
Update model information

#### DELETE /api/models/{model_name}
Delete model

### Inference Endpoints

#### POST /api/inference/single
Run inference on a single medical image

**Form Data:**
- `image`: Image file
- `model_name`: Name of model to use
- `confidence_threshold`: Detection confidence threshold (0.0-1.0, default: 0.25)
- `iou_threshold`: IoU threshold for NMS (0.0-1.0, default: 0.45)
- `return_image`: Return annotated image (boolean, default: true)

**Response:**
```json
{
  "success": true,
  "inference_type": "detection|classification|segmentation|regression",
  "model_name": "string",
  "model_domain": "string",
  "detections": [
    {
      "x1": 10.5,
      "y1": 20.3,
      "x2": 100.2,
      "y2": 150.7,
      "confidence": 0.95,
      "class_id": 0,
      "label": "polyp"
    }
  ],
  "classification": {
    "class_id": 0,
    "label": "Malignant",
    "confidence": 0.87,
    "probabilities": {
      "Benign": 0.13,
      "Malignant": 0.87
    }
  },
  "segmentation": {
    "mask": "data:image/png;base64,...",
    "classes_found": ["vessels", "lesions"],
    "pixel_counts": {
      "background": 50000,
      "vessels": 1500,
      "lesions": 300
    }
  },
  "regression": {
    "predicted_value": 14.5,
    "confidence_interval": {
      "lower": 13.5,
      "upper": 15.5
    },
    "unit": "years"
  },
  "annotated_image": "data:image/png;base64,...",
  "processing_time": 0.245,
  "image_info": {
    "width": 512,
    "height": 512,
    "mode": "RGB"
  }
}
```

#### POST /api/inference/batch
Run inference on multiple medical images

**Form Data:**
- `images`: Multiple image files (max 20)
- `model_name`: Name of model to use
- `confidence_threshold`: Detection confidence threshold
- `iou_threshold`: IoU threshold for NMS
- `return_images`: Return annotated images (boolean, default: false)

#### GET /api/inference/history
Get user's inference history

**Query Parameters:**
- `limit`: Number of records to return (default: 50)
- `offset`: Number of records to skip (default: 0)
- `model_name`: Filter by model name (optional)

#### GET /api/inference/statistics
Get user's inference statistics

#### GET /api/inference/models/available
Get list of available models for inference

## Medical Domains

The system supports the following medical domains:

1. **Mammography**: Breast cancer classification from mammography images
2. **Bone Age**: Age estimation from hand X-ray images
3. **Retina**: Retinal vessel and lesion segmentation
4. **Liver**: Liver tumor detection in CT scans
5. **Colon**: Polyp detection in colonoscopy images
6. **Brain**: Brain lesion detection in MRI scans
7. **MS Segmentation**: Multiple sclerosis lesion segmentation

## Model Types

### YOLO Models
- **Format**: `.pt`
- **Tasks**: Object detection
- **Domains**: Liver, colon, brain
- **Output**: Bounding boxes with confidence scores

### TensorFlow Models
- **Format**: `.h5`
- **Tasks**: Classification
- **Domains**: Mammography
- **Output**: Class probabilities

### PyTorch Models
- **Format**: `.pth`
- **Tasks**: Regression, segmentation
- **Domains**: Bone age, retina, MS segmentation
- **Output**: Numerical values or segmentation masks

## Error Codes

- `400`: Bad Request - Invalid input parameters
- `401`: Unauthorized - Authentication required
- `403`: Forbidden - Insufficient permissions
- `404`: Not Found - Resource not found
- `500`: Internal Server Error - Server error

## Example Usage

### 1. Login
```bash
curl -X POST "http://localhost:8001/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'
```

### 2. List Models
```bash
curl -X GET "http://localhost:8001/api/models/" \
  -H "Authorization: Bearer <token>"
```

### 3. Upload Model
```bash
curl -X POST "http://localhost:8001/api/models/upload" \
  -H "Authorization: Bearer <token>" \
  -F "file=@model.pt" \
  -F "name=liver_detector" \
  -F "model_type=yolo" \
  -F "task_type=detection" \
  -F "medical_domain=liver"
```

### 4. Run Inference
```bash
curl -X POST "http://localhost:8001/api/inference/single" \
  -H "Authorization: Bearer <token>" \
  -F "image=@medical_image.jpg" \
  -F "model_name=liver_detector" \
  -F "confidence_threshold=0.5"
```

## Development

### Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Set up MongoDB database
3. Configure environment variables
4. Run: `python main.py`

### Testing
Run unit tests with: `python -m pytest tests/`

### API Documentation
Interactive API documentation available at: `http://localhost:8001/api/docs`

## Support

For technical support or questions, please contact the development team or check the system logs for detailed error information.