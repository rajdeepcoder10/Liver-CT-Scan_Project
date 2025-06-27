<!-- Primary Meta Tags -->
<title>GitHub - rajdeepcoder10/Liver-CT-Scan_Project: This repository contains all the files, code and dataset related to the liver repository</title>
<meta name="title" content="GitHub - rajdeepcoder10/Liver-CT-Scan_Project: This repository contains all the files, code and dataset related to the liver repository" />
<meta name="description" content="This repository contains all the files, code and dataset related to the liver repository  - GitHub - rajdeepcoder10/Liver-CT-Scan_Project: This repository contains all the files, code and dataset related to the liver repository" />

<!-- Open Graph / Facebook -->
<meta property="og:type" content="website" />
<meta property="og:url" content="https://github.com/rajdeepcoder10/Liver-CT-Scan_Project" />
<meta property="og:title" content="GitHub - rajdeepcoder10/Liver-CT-Scan_Project: This repository contains all the files, code and dataset related to the liver repository" />
<meta property="og:description" content="This repository contains all the files, code and dataset related to the liver repository  - GitHub - rajdeepcoder10/Liver-CT-Scan_Project: This repository contains all the files, code and dataset related to the liver repository" />
<meta property="og:image" content="https://metatags.io/images/meta-tags.png" />

<!-- Twitter -->
<meta property="twitter:card" content="summary_large_image" />
<meta property="twitter:url" content="https://github.com/rajdeepcoder10/Liver-CT-Scan_Project" />
<meta property="twitter:title" content="GitHub - rajdeepcoder10/Liver-CT-Scan_Project: This repository contains all the files, code and dataset related to the liver repository" />
<meta property="twitter:description" content="This repository contains all the files, code and dataset related to the liver repository  - GitHub - rajdeepcoder10/Liver-CT-Scan_Project: This repository contains all the files, code and dataset related to the liver repository" />
<meta property="twitter:image" content="https://metatags.io/images/meta-tags.png" />

<!-- Meta Tags Generated with https://metatags.io -->

![Liver CT Scan Sample](https://raw.githubusercontent.com/rajdeepcoder10/Liver-CT-Scan_Project/main/Results/Screenshot%202025-06-16%20210718.png)

![image](https://github.com/user-attachments/assets/c7db11a0-9066-4d10-9945-a1aa6162aaf8)


This project uses VIT and VGG16 for Liver CT Scan image classification.

🩺 Liver Disease Detection using VGG16 and Vision Transformer (ViT)
🎯 A Deep Learning-powered system for automatic liver disease detection from CT scan images, using state-of-the-art VGG16 and ViT architectures.

📌 Project Overview
This project implements a web-based application that accepts CT-scan images and predicts the type of liver disease present using two powerful deep learning models:
✅ VGG16 – for feature-rich CNN-based detection
✅ Vision Transformer (ViT) – for advanced attention-based analysis

🧪 Detected Disease Categories
The system can classify images into the following categories:

🧬 Carcinoma

🧬 Squamous Cell Carcinoma

🧬 Adenocarcinoma

✅ Normal

🚀 Features
📤 Upload CT Scan image (PNG/JPG)

🧠 Model prediction using pretrained VGG16 / ViT

📊 Disease confidence score

📄 Auto-generated diagnosis report with treatment suggestions

🔥 Heatmap visualization (CAM)

🔐 Secure login system for access control

🛠️ Technologies Used
Tech Stack	Description
🧠 Deep Learning	VGG16, ViT (Vision Transformer)
🐍 Python	Core programming language
⚙️ Flask	Lightweight web framework
🎨 HTML/CSS	For web interface (Bootstrap 5 styled)
🖼️ OpenCV / PIL	Image pre-processing
🧪 TensorFlow/Keras	For building and loading DL models

📂 Folder Structure
csharp
Copy
Edit
.
├── Models/                 # Saved VGG16 and ViT models
├── uploads/               # Uploaded CT scan images
├── templates/
│   ├── index.html         # Upload & results dashboard
│   ├── login.html         # Secure login page
│   ├── report.html        # Diagnosis report page
│   └── heat.html          # Heatmap page
├── static/                # CSS / Images (if needed)
├── main.py                # Flask backend
├── README.md              # You're reading it!
💻 How to Run Locally
Clone the repository

bash
Copy
Edit
git clone https://github.com/your-username/liver-detection-vgg16-vit.git
cd liver-detection-vgg16-vit
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Place your trained model in the Models/ folder as model.h5

Run the server

bash
Copy
Edit
python main.py
Visit the app
Navigate to http://127.0.0.1:5000 in your browser.

📊 Example Output
✅ Predicted Disease: Adenocarcinoma

📈 Confidence Score: 91.24%

🩺 Recommendations:

Immunotherapy in conjunction with chemotherapy

Monitor tumor markers like AFP regularly

Liver transplant evaluation if condition worsens

🧠 About the Models
🔹 VGG16
16-layer CNN pre-trained on ImageNet

Fine-tuned on liver CT scan dataset

Excellent at extracting visual features like edges, texture, and shape

🔹 Vision Transformer (ViT)
Breaks image into patches and uses Transformer encoders

Learns spatial and semantic features via attention

Offers higher performance in some medical tasks due to its long-range attention

🌍 Real-World Applications
🏥 Assist doctors in early liver disease diagnosis

🔍 Reduce human error in medical image analysis

📄 Automatic report generation for patient use

📊 Medical AI research and education

🔐 Login Credentials (for demo)
Username: admin

Password: rajdeep

🧾 License
This project is licensed under the MIT License.
Feel free to fork and enhance it for academic or non-commercial purposes.

🤝 Acknowledgements
Kaggle – For dataset resources

Keras Applications – For pretrained models







