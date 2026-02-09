import re

def clean_text(text: str) -> str:
    """
    Fonction de nettoyage simple pour remplacer le fichier manquant.
    Enlève les caractères spéciaux basiques et les espaces en trop.
    """
    if not isinstance(text, str):
        return ""
    
    # Suppression basique du HTML et des retours à la ligne
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace('\n', ' ').strip()
    return text

def deduplicate(df):
    """
    Fonction dummy pour la déduplication.
    """
    return df