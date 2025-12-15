from colorama import Fore
import requests


def get_species_info(species_name):
    """
    Fetch the GBIF api to get species info
    """
    
    search_url = "https://api.gbif.org/v1/species/search"
    params = {'q': species_name}
    headers = {'User-Agent': 'Insect-ID-Script/1.0'}
    response = requests.get(search_url, params=params, headers=headers)
    
    if response.status_code != 200:
        return  Fore.RED + "Erreur lors de la requête API"
    

    data = response.json()
    # print(data)
    if not data['results']:
        return Fore.RED + "Aucune espèce trouvée pour ce nom"
    
    species_id = data['results'][0]['key']
    
    species_url =  Fore.BLUE + f"https://api.gbif.org/v1/species/{species_id}"
    species_response = requests.get(species_url, headers=headers)
    
    if species_response.status_code != 200:
        return  Fore.RED + "Erreur lors de la récupération des détails"
    
    species_data = species_response.json()
    
    info = {
        "nom_scientifique": species_data.get("scientificName"),
        "nom_commun": species_data.get("vernacularName"),
        "rang_taxonomique": species_data.get("rank"),
        "phylum": species_data.get("phylum"),
        "classe": species_data.get("class"),
        "ordre": species_data.get("order"),
        "famille": species_data.get("family"),
        "genre": species_data.get("genus"),
        "espece": species_data.get("species"),
        "description": species_data.get("description"),
        "distribution": species_data.get("distribution"),
        "references": species_data.get("references"),
        "url_gbif": f"https://www.gbif.org/species/{species_id}"
    }
    
    return Fore.LIGHTGREEN_EX + info
