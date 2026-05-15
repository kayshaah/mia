## Installation & Execution
This project requires Python 3.10+ and was built using standard machine learning libraries.

Setup the Environment:
Open your terminal (or Anaconda Prompt) and navigate to the unzipped project directory:

Run -> cd path/MIA_Project

*HERE path refers to your unzipped location
*We recommend creating a fresh virtual environment (e.g., conda create -n mia_project python=3.10)*

Run -> pip install torch torchvision torchaudio numpy scikit-learn matplotlib streamlit joblib pillow
Run -> streamlit run app.py

*THIS RUNS AS PER OUR CIFAR-10 DATASET

**** TO RUN FROM SCRATCH (Involves Training the model ~30/45 mins process)
Copy dataset_prep.py, training_model.py, shadow_model.py, attack_model.py, app.py in a new folder. 
Open Anaconda Script 

Run -> cd path/MIA_Project
Run -> pip install torch torchvision torchaudio numpy scikit-learn matplotlib streamlit joblib pillow
Run -> python dataset_prep.py 
Run -> python training_model.py 
Run -> python shadow_model.py 
Run -> python attack_model.py 
Run -> streamlit run app.py

Our Hardware - (7800X3D,7900XT) AMD CPU & GPU running on ROCm. Nvidia execution would be faster. 

You can additionally view the report and screenshots in the directory. 
