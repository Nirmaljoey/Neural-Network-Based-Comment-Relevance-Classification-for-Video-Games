# Neural-Network-Based-Comment-Relevance-Classification for Video Games

This repository contains the code for a machine learning project focused on classifying comments related to video games as relevant or irrelevant to the game itself. This project utilizes advanced deep learning techniques, specifically leveraging Transformer models, to achieve high accuracy in comment classification.

**Project Goals:**

* **Develop a robust and accurate comment classification model:** Train a deep learning model that can effectively distinguish between comments that are relevant to a specific video game (e.g., discussions about gameplay, story, graphics) and those that are irrelevant (e.g., off-topic discussions, spam, personal anecdotes).
* **Explore the effectiveness of Transformer models:** Investigate the performance of Transformer architectures, such as BERT, for this specific natural language processing task.
* **Fine-tune pre-trained models:** Leverage the power of pre-trained language models and fine-tune them on a dataset of video game comments to achieve optimal performance.
* **Build a user-friendly interface:** Develop a simple interface (e.g., a web application) for users to submit comments and receive real-time relevance predictions.

**Methodology:**

1. **Data Collection and Preprocessing:**
   - Gather a dataset of video game comments, including both relevant and irrelevant examples.
   - Clean and preprocess the data, including:
      - Removing noise (e.g., HTML tags, special characters).
      - Handling missing values.
      - Tokenization and lowercasing.
   - Create train, validation, and test sets.

2. **Model Development:**
   - **Experiment with different Transformer architectures:** 
      - BERT (Bidirectional Encoder Representations from Transformers)
      - ALBERT (A Lite BERT)
      - DistilBERT (Distilled BERT) 
   - **Fine-tune pre-trained models:** Adjust the weights of the pre-trained models on the video game comment dataset.
   - **Implement and train classification models:** Utilize the fine-tuned models for comment relevance classification.

3. **Model Evaluation:**
   - Evaluate model performance using metrics such as accuracy, precision, recall, F1-score, and AUC.
   - Analyze model predictions and identify areas for improvement.

4. **Deployment (Optional):**
   - Develop a user interface (e.g., a web application) for users to submit comments.
   - Integrate the trained model into the interface to provide real-time predictions.

**Technologies Used:**

* **Python:** Primary programming language.
* **Transformers library:** For working with Transformer models (Hugging Face).
* **TensorFlow/PyTorch:** Deep learning frameworks.
* **Scikit-learn:** For data preprocessing and model evaluation.
* **Streamlit (Optional):** For building a simple web application.

**Project Structure:**
