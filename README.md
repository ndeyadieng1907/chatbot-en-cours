Chatbot ANSD - Application de Question-Réponse Basée sur les Publications de l'ANSD
Description
Le Chatbot ANSD est une application de question-réponse basée sur les publications de l'Agence Nationale de la Statistique et de la Démographie (ANSD). L'application permet aux utilisateurs de dialoguer avec les documents statistiques de l'ANSD pour obtenir des informations précises et pertinentes. En intégrant des fonctionnalités textuelles et vocales, ce chatbot vise à simplifier l'accès à l'information statistique pour une meilleure prise de décision.

Pourquoi ce projet est utile
Accès simplifié à l'information : Les utilisateurs peuvent poser des questions et obtenir des réponses directes basées sur les documents de l'ANSD.
Multimodalité : L'application supporte les interactions textuelles et vocales, offrant ainsi une expérience utilisateur enrichie et plus flexible.
Utilisation de technologies avancées : Utilise des modèles de langage de pointe (GPT-3.5 Turbo), ainsi que des technologies de traitement du langage naturel (Langchain, LamaIndex).
Fonctionnalités
Téléchargement de fichiers : Supporte les formats PDF, DOCX et TXT.
Interrogation textuelle et vocale : Posez des questions par texte ou par voix.
Synthèse et transcription vocale : Utilise les API Google Cloud pour convertir la voix en texte et vice versa.
Base de données vectorielle : Utilise Langchain pour stocker et gérer les plongements des données textuelles.
Interface utilisateur réactive : Développée avec Streamlit pour des interactions en temps réel.
Prise en main
Prérequis
Python : Assurez-vous d'avoir Python installé (version 3.7 ou supérieure).
Clés API : Vous aurez besoin de clés API pour OpenAI et Google Cloud. Créez un projet Google Cloud, activez les API nécessaires et téléchargez le fichier JSON des informations d'identification.
Installation
Cloner le dépôt :

sh
Copy code
git clone https://github.com/votre-utilisateur/chatbot-ansd.git
cd chatbot-ansd
Créer un environnement virtuel :

sh
Copy code
python -m venv venv
Activer l'environnement virtuel :

Windows :
sh
Copy code
source venv/Scripts/activate
Mac/Linux :
sh
Copy code
source venv/bin/activate
Installer les dépendances :

sh
Copy code
pip install -r requirements.txt
Configurer les variables d'environnement :
Créez un fichier .env à la racine du projet et ajoutez vos clés API :

plaintext
Copy code
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/google-cloud-key.json
OPENAI_API_KEY=your_openai_api_key
Lancement de l'application
sh
Copy code
streamlit run app.py
Où obtenir de l'aide
Si vous avez des questions ou des problèmes, vous pouvez :

Consulter la documentation dans le dépôt.
Ouvrir une issue sur GitHub.
Contacter les mainteneurs du projet.
Mainteneurs et contributeurs
Nom du Mainteneur - GitHub Profile
Contributeurs - Voir la liste des contributeurs qui participent à ce projet en visitant contributors.
