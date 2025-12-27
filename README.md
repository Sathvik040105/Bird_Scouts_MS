## Bird Scouts

Bird Scouts is a web application created using streamlit to help nature enthusiasts and researchers identify bird species through image or audio inputs.
It also includes a tree species identification feature and a collaborative community map for bird sightings, creating an engaging way to explore and contribute to biodiversity knowledge.

 
This project is part of *UMC 301 - Applied Data Science and Artificial Intelligence* course at IISc. **[Presentation slides](https://docs.google.com/presentation/d/1AmpiQddBaHowNwmOWfrJYQHjGgEAjgNN2WpIVIsTEwI/edit#slide=id.g3176ed11d7c_1_21)**


https://github.com/user-attachments/assets/2128beef-d2aa-4c4e-8204-81ab82dabe53



### Features 

+ Bird Species Identification
    - Upload an image or audio file of a bird.
    - Get instant predictions on the bird species and call types.

+ Tree Species Identification
    - Identify tree species by uploading images of trunks or leaves.

+ Community Map for Bird Sightings
    - Share and explore bird sightings on an interactive map.
    - Collaborate with other users to track bird populations and habitats across regions.


### Setup

#### Prerequisites
- Python 3.10 or higher
- pip (Python package installer)

#### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/Bird_Scouts_MS.git
   cd Bird_Scouts_MS
   ```

2. Create a virtual environment (recommended):
   ```bash
   python3 -m venv .venv
   ```

3. Activate the virtual environment:
   - On Windows: `.venv\Scripts\activate`
   - On macOS/Linux: `source .venv/bin/activate`

4. Install the required dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

#### Running Locally
To run the application locally:
```bash
streamlit run main.py
```
This will start the Streamlit web app on your local machine (usually at `http://localhost:8501`).

#### Training Models
To retrain the models, you will need the appropriate datasets (the original training datasets are not included in this repository due to size constraints). The training code is provided in the `training_files/` directory.

Navigate to the `training_files/` directory and run the respective Jupyter notebooks or Python scripts. Ensure you have Jupyter Notebook installed (`pip install jupyter` if needed).

- Bird call classification: Run `jupyter notebook birdcalltraining_audio.ipynb`
- Species identification from audio: Run `jupyter notebook species_identification_audio.ipynb`
- Bird image classification: Run `jupyter notebook Bird_image_AdityaM.ipynb`
- Feather image classification: Run `jupyter notebook feather_AdityaM.ipynb`
- Bark/trunk image classification: Run `python bark_MobileNetv3.py`
- Leaf image classification: Run `python leaves.py`

Note: Training requires significant computational resources (preferably with GPU support for PyTorch models). You may need to adapt the data paths in the scripts to point to your local datasets. The notebooks were originally designed for Kaggle environments with specific dataset paths.

### Contributions

| Name | Contribution | GitHub Profile |
|---| --- | --- |
|Aditya| Bird species identifier with bird image/ feather image | [Aditya-Manjunatha](https://github.com/Aditya-Manjunatha)
|Krishna| Dataset handling, Web Scraping | [ELNKrishna](https://github.com/ELNKrishna)
|Nagasai| Backend, Website-Logic, LLM, RAG | [Nagasai561](https://github.com/Nagasai561)
|Sanyat| Tree species identifier with trunk/leaf image, BBox detection | [SanyatFale](https://github.com/SanyatFale)   
|Sathvik| Species identifier with audio, Flowcharts | [sathvikb007](https://github.com/Sathvik040105)
|Shankar| Type of sound identifier, Website-UI, User-auth | [OmegaSun18](https://github.com/OmegaSun18)
