# Import necessary libraries
import requests
import os
from bs4 import BeautifulSoup
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
import newspaper
from newspaper import Article
import nltk
import statistics
import collections
from nltk.collocations import TrigramAssocMeasures, TrigramCollocationFinder
from nltk.collocations import QuadgramAssocMeasures, QuadgramCollocationFinder
import time
from openai import OpenAI
#client = OpenAI(api_key="")
import pandas as pd
import re
import streamlit as st
from apify_client import ApifyClient
import pandas as pd
import transformers
from transformers import GPT2Tokenizer

import json
#openai.api_key = openai.api_key = os.environ['openai_api_key']
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')
nltk.download('words')
nltk.download('maxent_ne_chunker')
nltk.download('vader_lexicon')
nltk.download('inaugural')
nltk.download('webtext')
nltk.download('treebank')
nltk.download('gutenberg')
nltk.download('genesis')
nltk.download('trigram_collocations')
nltk.download('quadgram_collocations')


# Define a function to scrape Google search results and create a dataframe
from apify_client import ApifyClient
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def scrape_google(search):
    # Define the Apify API URL and the actor's name
    APIFY_API_URL = 'https://api.apify.com/v2'
    ACTOR_NAME = 'apify/google-search-scraper'

    # Retrieve the Apify API key from Streamlit secrets
    APIFY_API_KEY = 'apify_api_L01dAtczculILhxDMjpfCKYskPS7iJ2HQKTO'

    # Initialize the ApifyClient with your API token
    client = ApifyClient(APIFY_API_KEY)

    # Prepare the actor input
    run_input = {
        "countryCode": "be",
        "csvFriendlyOutput": False,
        "customDataFunction": "async ({ input, $, request, response, html }) => {\n  return {\n    pageTitle: $('title').text(),\n  };\n};",
        "includeUnfilteredResults": False,
        "languageCode": "fr",
        "maxPagesPerQuery": 1,
        "mobileResults": False,
        "queries": search,
        "resultsPerPage": 10,
        "saveHtml": False,
        "saveHtmlToKeyValueStore": False
    }

    print(f"Running Google Search Scrape for {search}")
    # Run the actor and wait for it to finish
    run = client.actor(ACTOR_NAME).call(run_input=run_input)
    print(f"Finished Google Search Scrape for {search}")

    # Fetch the actor results from the run's dataset
    results = []
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        results.append(item)

    # Extract URLs from organic results
    organic_results = [item['organicResults'] for item in results]
    urls = [result['url'] for sublist in organic_results for result in sublist]

    # Create DataFrame
    df = pd.DataFrame(urls, columns=['url'])

    # Print the dataframe
    print(df)
    st.header("Scraped Data from SERP and SERP Links")
    return df



@st.cache_data(show_spinner=False)
def scrape_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except:
        return ""



@st.cache_data(show_spinner=False)
def truncate_to_token_length(input_string, max_tokens=1700):
    # Tokenize the input string
    tokens = tokenizer.tokenize(input_string)
    
    # Truncate the tokens to a maximum of max_tokens
    truncated_tokens = tokens[:max_tokens]
    
    # Convert the truncated tokens back to a string
    truncated_string = tokenizer.convert_tokens_to_string(truncated_tokens)
    
    return truncated_string


# Define a function to perform NLP analysis and return a string of keyness results

@st.cache_data(show_spinner=False)
def analyze_text(text):
    # Tokenize the text and remove stop words
    tokens = [word.lower() for word in word_tokenize(text) if word.isalpha() and word.lower() not in stopwords.words('french')]
    # Get the frequency distribution of the tokens
    fdist = FreqDist(tokens)
    # Create a bigram finder and get the top 20 bigrams by keyness
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(tokens)
    bigrams = finder.nbest(bigram_measures.raw_freq, 20)
    # Create a string from the keyness results
    results_str = ''
    results_str += 'Top 20 Words:\n'
    for word, freq in fdist.most_common(20):
        results_str += f'{word}: {freq}\n'
    results_str += '\nTop 20 Bigrams:\n'
    for bigram in bigrams:
        results_str += f'{bigram[0]} {bigram[1]}\n'
    st.write(results_str)    
    return results_str

# Define the main function to scrape Google search results and analyze the article text

@st.cache_data(show_spinner=False)
def main(query):
    # Scrape Google search results and create a dataframe
    df = scrape_google(query)
    # Scrape article text for each search result and store it in the dataframe
    for index, row in df.iterrows():
        url = row['URL']
        article_text = scrape_article(url)
        df.at[index, 'Article Text'] = article_text
    # Analyze the article text for each search result and store the keyness results in the dataframe
    for index, row in df.iterrows():
        text = row['Article Text']
        keyness_results = analyze_text(text)
        df.at[index, 'Keyness Results'] = keyness_results
    # Return the final dataframe
    return df



# Define the main function to scrape Google search results and analyze the article text

@st.cache_data(show_spinner=False)
def analyze_serps(query):
    # Scrape Google search results and create a dataframe
    df = scrape_google(query)
    # Scrape article text for each search result and store it in the dataframe
    for index, row in df.iterrows():
        url = row['url']
        #st.write(url)
        article_text = scrape_article(url)
        df.at[index, 'Article Text'] = article_text
    # Analyze the article text for each search result and store the NLP results in the dataframe
    for index, row in df.iterrows():
        text = row['Article Text']
        # Tokenize the text and remove stop words
        tokens = [word.lower() for word in word_tokenize(text) if word.isalpha() and word.lower() not in stopwords.words('french') and 'contact' not in word.lower() and 'admin' not in word.lower()]
        # Calculate the frequency distribution of the tokens
        fdist = FreqDist(tokens)
        # Calculate the 20 most common words
        most_common = fdist.most_common(20)
        # Calculate the 20 least common words
        least_common = fdist.most_common()[-20:]
        # Calculate the 20 most common bigrams
        bigram_measures = BigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(tokens)
        bigrams = finder.nbest(bigram_measures.raw_freq, 20)
        # Calculate the 20 most common trigrams
        trigram_measures = TrigramAssocMeasures()
        finder = TrigramCollocationFinder.from_words(tokens)
        trigrams = finder.nbest(trigram_measures.raw_freq, 20)
        # Calculate the 20 most common quadgrams
        quadgram_measures = QuadgramAssocMeasures()
        finder = QuadgramCollocationFinder.from_words(tokens)
        quadgrams = finder.nbest(quadgram_measures.raw_freq, 20)
        # Calculate the part-of-speech tags for the text
        pos_tags = nltk.pos_tag(tokens)
        # Store the NLP results in the dataframe
        df.at[index, "Facts"] = generate_content3(text)
        df.at[index, 'Most Common Words'] = ', '.join([word[0] for word in most_common])
        df.at[index, 'Least Common Words'] = ', '.join([word[0] for word in least_common])
        df.at[index, 'Most Common Bigrams'] = ', '.join([f'{bigram[0]} {bigram[1]}' for bigram in bigrams])
        df.at[index, 'Most Common Trigrams'] = ', '.join([f'{trigram[0]} {trigram[1]} {trigram[2]}' for trigram in trigrams])
        df.at[index, 'Most Common Quadgrams'] = ', '.join([f'{quadgram[0]} {quadgram[1]} {quadgram[2]} {quadgram[3]}' for quadgram in quadgrams])
        df.at[index, 'POS Tags'] = ', '.join([f'{token}/{tag}' for token, tag in pos_tags])
        # Replace any remaining commas with spaces in the Article Text column
        df.at[index, 'Article Text'] = ' '.join(row['Article Text'].replace(',', ' ').split())
    st.write(df)
    return df




# Define a function to summarize the NLP results from the dataframe


@st.cache_data(show_spinner=False)
def summarize_nlp(df):
    # Calculate the total number of search results
    total_results = len(df)
    # Calculate the average length of the article text
    avg_length = round(df['Article Text'].apply(len).mean(), 2)
    # Get the most common words across all search results
    all_words = ', '.join(df['Most Common Words'].sum().split(', '))
    # Get the most common bigrams across all search results
    all_bigrams = ', '.join(df['Most Common Bigrams'].sum().split(', '))
    # Get the most common trigrams across all search results
    all_trigrams = ', '.join(df['Most Common Trigrams'].sum().split(', '))
    # Get the most common quadgrams across all search results
    all_quadgrams = ', '.join(df['Most Common Quadgrams'].sum().split(', '))
    # Get the most common part-of-speech tags across all search results
    all_tags = ', '.join(df['POS Tags'].sum().split(', '))
    # Calculate the median number of words in the article text
    median_words = statistics.median(df['Article Text'].apply(lambda x: len(x.split())).tolist())
    # Calculate the frequency of each word across all search results
    word_freqs = collections.Counter(all_words.split(', '))
    # Calculate the frequency of each bigram across all search results
    bigram_freqs = collections.Counter(all_bigrams.split(', '))
    # Calculate the frequency of each trigram across all search results
    trigram_freqs = collections.Counter(all_trigrams.split(', '))
    # Calculate the frequency of each quadgram across all search results
    quadgram_freqs = collections.Counter(all_quadgrams.split(', '))
    # Calculate the top 20% of most frequent words
    top_words = ', '.join([word[0] for word in word_freqs.most_common(int(len(word_freqs) * 0.2))])
    # Calculate the top 20% of most frequent bigrams
    top_bigrams = ', '.join([bigram[0] for bigram in bigram_freqs.most_common(int(len(bigram_freqs) * 0.2))])
    # Calculate the top 20% of most frequent trigrams
    top_trigrams = ', '.join([trigram[0] for trigram in trigram_freqs.most_common(int(len(trigram_freqs) * 0.2))])
    # Calculate the top 20% of most frequent quadgrams
    top_quadgrams = ', '.join([quadgram[0] for quadgram in quadgram_freqs.most_common(int(len(quadgram_freqs) * 0.2))])
    
    summary = ""
    summary += f'Résultats totaux : {total_results}\n'
    summary += f'Longueur moyenne des articles : {avg_length} characters\n'
    summary += f'Nombre médian de mots par article : {median_words}\n'
    summary += f'Most common words: {top_words} ({len(word_freqs)} total words)\n'
    summary += f'Most common bigrams: {top_bigrams} ({len(bigram_freqs)} total bigrams)\n'
    summary += f'Most common trigrams: {top_trigrams} ({len(trigram_freqs)} total trigrams)\n'
    summary += f'Most common quadgrams: {top_quadgrams} ({len(quadgram_freqs)} total quadgrams)\n'
    return summary

@st.cache_data(show_spinner=False)
def generate_content(prompt, model="gpt-3.5-turbo", max_tokens=1000, temperature=0.4):
    prompt = truncate_to_token_length(prompt,2500)
    gpt_response = client.chat.completions.create(model=model,
    messages=[
        {"role": "system", "content": "Simulez un journaliste et un rédacteur en chef exceptionnellement talentueux. À partir des instructions suivantes, réfléchissez étape par étape et produisez le meilleur résultat possible."},
        {"role": "user", "content": prompt}],
    max_tokens=max_tokens,
    n=1,
    stop=None,
    temperature=temperature)
    response = gpt_response['choices'][0]['message']['content'].strip()
    response = response
    return response.strip().split('\n')

@st.cache_data(show_spinner=False)
def generate_content2(prompt, model="gpt-3.5-turbo", max_tokens=1000, temperature=0.4):
    prompt = truncate_to_token_length(prompt,2500)
    gpt_response = client.chat.completions.create(model=model,
    messages=[
        {"role": "system", "content": "Simulez un journaliste et un rédacteur en chef exceptionnellement talentueux. À partir des instructions suivantes, réfléchissez étape par étape et produisez le meilleur résultat possible. Renvoyez les résultats en format markdown, s'il vous plaît."},
        {"role": "user", "content": prompt}],
    max_tokens=max_tokens,
    n=1,
    stop=None,
    temperature=temperature)
    response = gpt_response['choices'][0]['message']['content'].strip()
    response = response
    return response
    
@st.cache_data(show_spinner=False)
def generate_content3(prompt, model="gpt-3.5-turbo", max_tokens=1000, temperature=0.4):
    prompt = truncate_to_token_length(prompt,2500)
    gpt_response = client.chat.completions.create(model=model,
    messages=[
        {"role": "system", "content": "Simulez un journaliste d'investigation et un chercheur exceptionnellement talentueux. À partir du texte suivant, rédigez un court paragraphe ne reprenant que les faits les plus importants et les éléments à retenir qui pourront être utilisés ultérieurement lors de la rédaction d'une analyse ou d'un article complet."},
        {"role": "user", "content": f"Utilisez le texte suivant pour fournir la lecture : {prompt}"}],
    max_tokens=max_tokens,
    n=1,
    stop=None,
    temperature=temperature)
    response = gpt_response['choices'][0]['message']['content'].strip()
    response = response
    return response       
    
@st.cache_data(show_spinner=False)
def generate_semantic_improvements_guide(prompt,query, model="gpt-3.5-turbo", max_tokens=2000, temperature=0.4):
    prompt = truncate_to_token_length(prompt,1500)
    gpt_response = client.chat.completions.create(model=model,
    messages=[
        {"role": "system", "content": """Vous êtes un expert en référencement sémantique. En particulier, vous êtes surhumain quand il s'agit de prendre un rapport NLTK donné sur un corpus de texte donné compilé à partir du texte des pages liées renvoyées par une recherche Google.
        et à l'utiliser pour élaborer un ensemble complet d'instructions à l'intention d'un rédacteur d'article qui peut être utilisé pour informer quelqu'un qui écrit un article long sur un sujet donné afin qu'il puisse couvrir au mieux le référencement sémantique tel qu'il apparaît dans les données NLTK du corpus SERP. 
        Fournir le résultat dans un format markdown bien formaté. Le but de ce guide est d'aider le rédacteur à s'assurer que le contenu qu'il crée est aussi complet que possible pour le référencement sémantique, en mettant l'accent sur ce qui est le plus important du point de vue du référencement sémantique."""},
        {"role": "user", "content": f"Données de référencement sémantique pour le mot-clé basé sur le contenu qui se classe sur la première page de Google pour la requête de mot-clé donnée : {query} et les données sémantiques qui s'y rapportent :  {prompt}"}],
    max_tokens=max_tokens,
    n=1,
    stop=None,
    temperature=temperature)
    response = gpt_response['choices'][0]['message']['content'].strip()
    st.header("Semantic Improvements Guide")
    st.markdown(response,unsafe_allow_html=True)
    return str(response) 
   
@st.cache_data(show_spinner=False)
def generate_outline(topic, model="gpt-3.5-turbo", max_tokens=1500):
    prompt = f"Créez un plan d'article très complet pour le sujet : {topic}. Envisagez tous les angles possibles et soyez aussi exhaustif que possible. Veuillez utiliser des chiffres romains pour chaque section."
    outline = generate_content(prompt, model=model, max_tokens=max_tokens)
    return outline

@st.cache_data(show_spinner=False)
def improve_outline(outline, semantic_readout, model="gpt-3.5-turbo", max_tokens=1500):
    prompt = f"A partir du plan d'article suivant, veuillez l'améliorer et l'étendre autant que possible en gardant à l'esprit les mots-clés SEO et les données fournies dans notre lecture sémantique. N'incluez pas de section sur le référencement sémantique lui-même, vous utilisez la lecture pour mieux informer votre création de l'ébauche. Essayez de l'inclure et de l'étendre autant que possible. Veuillez utiliser des chiffres romains pour chaque section. L'objectif est d'obtenir un aperçu aussi complet, clair et utile que possible, en explorant le sujet de manière aussi approfondie que possible. Réfléchissez étape par étape avant de répondre. Veuillez prendre en considération la lecture sémantique du référencement fournie ici : {semantic_readout} qui devrait vous aider à déterminer certaines des améliorations que vous pouvez apporter, bien que vous puissiez également envisager des améliorations supplémentaires qui ne sont pas incluses dans cette lecture sémantique du référencement.  Schéma à améliorer : {outline}."
    improved_outline = generate_content(prompt, model=model, max_tokens=max_tokens)
    return improved_outline

@st.cache_data(show_spinner=False)
def generate_sections(improved_outline, model="gpt-3.5-turbo", max_tokens=2000):
    sections = []
    # Analyser le plan pour identifier les principales sections
    major_sections = []
    current_section = []
    for part in improved_outline:
        if re.match(r'^[ \t]*[#]*[ \t]*(I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV)\b', part):
            if current_section:  # not the first section
                major_sections.append('\n'.join(current_section))
                current_section = []
        current_section.append(part)
    if current_section:  # Append the last section
        major_sections.append('\n'.join(current_section))

    # Générer du contenu pour chaque grande section
    for i, section_outline in enumerate(major_sections):
        full_outline = "Le schéma complet amélioré est donné : "
        full_outline += '\n'.join(improved_outline)
        specific_section = "et en se concentrant plus particulièrement sur la section suivante : "
        specific_section += section_outline
        prompt =  specific_section + ", veuillez rédiger une section complète qui va en profondeur, fournit des détails et des preuves, et ajoute autant de valeur supplémentaire que possible. Conservez toute hiérarchie que vous trouvez. Ne rédigez jamais la conclusion d'une section à moins que la section elle-même ne soit censée être une conclusion. Texte de la section :"
        section = generate_content(prompt, model=model, max_tokens=max_tokens)
        sections.append(section)
    return sections

@st.cache_data(show_spinner=False)
def improve_section(section, i, model="gpt-3.5-turbo", max_tokens=1500):
    prompt = f"Étant donné la section suivante de l'article : {section}, veuillez apporter des améliorations à cette section. Conservez toute hiérarchie que vous trouvez. Fournissez uniquement la section mise à jour, pas le texte de votre recommandation, faites simplement les changements. Fournissez toujours la section mise à jour en Markdown valide s'il vous plaît. Section mise à jour avec améliorations :"
    prompt = str(prompt)
    improved_section = generate_content2(prompt, model=model, max_tokens=max_tokens)
    st.markdown(improved_section,unsafe_allow_html=True)
    return " ".join(improved_section)  # join the lines into a single string

@st.cache_data(show_spinner=False)
def concatenate_files(file_names, output_file_name):
    final_draft = ''
    
    for file_name in file_names:
        with open(file_name, 'r') as file:
            final_draft += file.read() + "\n\n"  # Add two newline characters between sections

    with open(output_file_name, 'w') as output_file:
        output_file.write(final_draft)
    return final_draft


@st.cache_data(show_spinner=False)
def generate_article(topic, model="gpt-3.5-turbo", max_tokens_outline=2000, max_tokens_section=2000, max_tokens_improve_section=4000):
    status = st.empty()
    status.text('Analyse SERPs...')
    
    query = topic
    results = analyze_serps(query)
    summary = summarize_nlp(results)

    status.text('Générer une lecture sémantique du SEO...')
    semantic_readout = generate_semantic_improvements_guide(topic, summary,  model=model, max_tokens=max_tokens_outline)
    
    
    status.text('Création d une première ébauche...')
    initial_outline = generate_outline(topic, model=model, max_tokens=max_tokens_outline)

    status.text('Améliorer l ébauche initiale...')
    improved_outline = improve_outline(initial_outline, semantic_readout, model=model, max_tokens=1500)
    #st.markdown(improved_outline,unsafe_allow_html=True)
    
    status.text('Générer des sections sur la base du schéma amélioré...')
    sections = generate_sections(improved_outline, model=model, max_tokens=max_tokens_section)

    status.text('Amélioration des sections...')
    
    improved_sections = []
    for i, section in enumerate(sections):
        section_string = '\n'.join(section)
        status.text(f'Improving section {i+1} of {len(sections)}...')
        time.sleep(5)
        improved_sections.append(improve_section(section_string, i, model=model, max_tokens=1200))

    status.text('Fini')
    final_content = '\n'.join(improved_sections)
 
def main():
    st.title('Générateur d articles longs avec compréhension du référencement sémantique')
    
    st.markdown('''
    Bienvenue dans le générateur d'articles longs ! Cette application s'appuie sur une IA avancée pour créer des articles complets basés sur le sujet que vous lui fournissez. 

    Non seulement elle génère des articles, mais elle inclut également une compréhension du référencement sémantique.

    Saisissez simplement votre sujet ci-dessous et laissez l'IA faire sa magie !
    ''')
   
    topic = st.text_input("Enter topic:", "Acheter en 2050")

    # Get user input for API key
    client = st.text_input("Entrez votre clé API OpenAI")
    if st.button('Generate Content'):
        if client:
            with st.spinner("Generating content..."):
                final_draft = generate_article(topic)
        else:
            st.warning("Please enter your OpenAI API key above.")

if __name__ == "__main__":
    main()
