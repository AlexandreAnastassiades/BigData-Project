import pandas as pd
import category_encoders as ce
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import re


def main():
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

    # To clean text
    def preprocess_text(sentence):
        sentence = str(sentence)
        # Lowercase text
        sentence = sentence.lower()
        # Remove whitespace
        sentence = sentence.replace('{html}', "")
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', sentence)
        # Remove weblinks
        rem_url = re.sub(r'http\S+', '', cleantext)
        # Remove numbers
        rem_num = re.sub('[0-9]+', '', rem_url)
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(rem_num)
        # Remove StopWords
        filtered_words = [w for w in tokens if len(w) > 2 if
                          not w in stopwords.words('french') or not w in stopwords.words('english')]
        return " ".join(filtered_words)

    # Load data
    airbnb = pd.read_csv('33000-BORDEAUX_nettoye.csv')
    columns_to_remove = ['Identifiant',
                         'Url',
                         'Titre',
                         'Resume',
                         'NbLits',
                         'NombreSdB',
                         'type_lit',
                         'Animal_sur_place',
                         'reglement_interieur',
                         'prix_nuitee']
    # Removing some missing data
    airbnb = airbnb[
        (airbnb["PrixNuitee"] > 0) &
        (airbnb["Resume"].notna()) &
        (airbnb["Description"].notna()) &
        (airbnb["type_propriete"] != "Inconnue")]

    airbnb = airbnb.drop(columns=columns_to_remove)
    columns_name = {
        "PrixNuitee": "price",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "Capacite_accueil": "max_people_count",
        "NombreSdB": "bathroom_count",
        "NbChambres": "bedroom_count",
        "Type_logement": "housing_type",
        "type_propriete": "property_type",
        "Cuisine": "kitchen",
        "Internet": "internet",
        "television": "television",
        "produits_base": "commodities",
        "Shampooing": "shampooing",
        "Chauffage": "heating",
        "Climatisation": "air_conditioning",
        "machine_laver": "washing_machine",
        "seche_linge": "dryer",
        "parking_sur-place": "parking",
        "wifi": "wifi",
        "television_cable": "tv_decoder",
        "petit_dejeuner": "breakfast",
        "animaux_acceptes": "pets_allowed",
        "pourEnfants_famille": "for_kids",
        "adapte_evenements": "adapted_for_events",
        "logement_fumeur": "smoking",
        "accessibilite": "accessibility",
        "Ascenseur": "elevator",
        "cheminee_interieur": "fireplace",
        "Interphone": "intercom",
        "Portier": "doorman",
        "Piscine": "swimming_pool",
        "Jacuzzi": "jacuzzi",
        "salle_sport": "gym",
        "Entree_24-24": "24_hours_entry",
        "Cintres": "hangers",
        "fer_repasser": "iron",
        "seche_cheveux": "hair_dryer",
        "espace_travail_ordi": "workspace",
        "detecteur_fumee": "smoke_detector",
        "monoxyde_carbone_detect": "carbon_monoxide_detector",
        "kit_secours": "rescue_kit",
        "fiche_securite": "safety_sheet",
        "extincteur": "extinguisher",
        "porte_chambre_verrou": "bedroom_lock",
        "rection_semaine": "discount_week",
        "reduction_mois": "discount_mounth",
        "surcout_voyageur_supp": "additional_traveler_cost",
        "frais_menage": "cleaning_cost",
        "Caution": "deposit",
        "conditions_annulation": "cancel_conditions",
        "Description": "description",
        "duree_minimale_sejour": "minimum_stay",
    }
    # Renaming the data frame
    airbnb = airbnb.rename(columns=columns_name)
    # Encoding housing_type column
    one_hot_encoder = ce.OneHotEncoder(cols=['housing_type'])
    housing_type_one_hot = one_hot_encoder.fit_transform(airbnb["housing_type"])
    airbnb = pd.concat([airbnb, housing_type_one_hot], axis=1)
    airbnb = airbnb.drop(columns=['housing_type'])
    # Encoding property_type column
    binary_encoder = ce.BinaryEncoder(cols=['property_type'])
    property_type_binary = binary_encoder.fit_transform(airbnb["property_type"])
    airbnb = pd.concat([airbnb, property_type_binary], axis=1)
    airbnb = airbnb.drop(columns=['property_type'])
    # Encoding cancel_conditions column
    cancel_conditions_order = {
        'None': 1,
        'Flexibles': 2,
        'Modérées': 3,
        'Strictes': 4
    }
    airbnb["cancel_conditions"] = airbnb["cancel_conditions"].fillna('None')
    airbnb['cancel_conditions'] = airbnb.cancel_conditions.map(cancel_conditions_order)

    airbnb['description_pre'] = airbnb_word['description'].map(lambda s: preprocess_text(s))
    # Adding description related features
    for feature in final_features:  ## Nom du fichier avec les mots
        for desc in airbnb_word_processing_tuto['description_pre']:
            if feature in desc:
                final_features[feature].append(1)
            else:
                final_features[feature].append(0)
    for feature in final_features:
        airbnb[feature] = final_features[feature]

    ## Predict du model

    ## Ajout du predict dans le df
    ## Convert CSV
    ## Fin
