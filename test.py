import pandas as pd
import requests
from bs4 import BeautifulSoup
from joblib import load
import urllib.parse
from serpapi import GoogleSearch

# Chargement du modèle et des features
model, x_train = load('diagnosis.joblib')

def make_prediction(model, feature_names):
    take_input = {}
    for f in feature_names:
        v = input(f'{f} [0: Non / 1: Oui] : ')
        take_input[f] = int(v)
    df_in = pd.DataFrame([take_input])
    feature_true = [k for k, v in take_input.items() if v == 1]
    prediction = model.predict(df_in)
    return prediction[0], feature_true

def build_query(symptoms, disease, site=None):
    base = " ".join(symptoms) + " symptoms " + disease
    if site:
        base += f" site:{site}"
    return urllib.parse.quote_plus(base)

def search_web_serpapi(query, num_results=5):
    params = {
        "q": query,
        "api_key": "8c6a979923b8abdc35eac9872eeeb696eb7e744d2de082891de0f64febb77cca",  # Clé API SerpAPI
        "num": num_results
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    links = []
    for result in results.get("organic_results", []):
        title = result.get("title")
        link = result.get("link")
        if title and link:
            links.append((title, link))
    return links

def fetch_snippet(url):
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        s = BeautifulSoup(r.text, 'html.parser')

        # On récupère uniquement les paragraphes longs
        paragraphs = s.find_all('p')
        texts = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 60]
        
        if texts:
            return "\n".join(texts[:3])  # 3 premiers paragraphes utiles
        else:
            return "[Aucun contenu médical significatif trouvé]"
    except Exception as e:
        return f"[Erreur de récupération: {str(e)}]"

if __name__ == "__main__":
    feature_names = x_train.columns
    disease, symptoms = make_prediction(model, feature_names)
    print(f"\n🧠 Diagnostic prédit : {disease}")
    print(f"🩺 Symptômes détectés : {symptoms}")

    query = build_query(symptoms, disease)
    print(f"\n🔍 Recherche en cours: \"{query}\" ...\n")
    results = search_web_serpapi(query, num_results=5)

    for i, (title, url) in enumerate(results, 1):
        print(f"{i}. 🔗 {title}\n   URL: {url}")
        snippet = fetch_snippet(url)
        print(f"   📄 Extrait :\n   {snippet}\n{'-'*60}")
