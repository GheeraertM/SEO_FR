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
import openai
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
        "languageCode": "nl",
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
    #st.write(df)
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
    tokens = [word.lower() for word in word_tokenize(text) if word.isalpha() and word.lower() not in stopwords.words('dutch')]
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
    #df.to_csv("NLP_Data_On_SERP_Links_Text.csv")
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
        tokens = [word.lower() for word in word_tokenize(text) if word.isalpha() and word.lower() not in stopwords.words('dutch') and 'contact' not in word.lower() and 'admin' not in word.lower()]
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
    # Save the final dataframe as an Excel file
    #writer = pd.ExcelWriter('NLP_Based_SERP_Results.xlsx', engine='xlsxwriter')
    #df.to_excel(writer, sheet_name='Sheet1', index=False)
    #writer.save()
    st.write(df)
    # Return the final dataframe
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
    
    #print(f'Total results: {total_results}')
    #print(f'Average article length: {avg_length} characters')
    #print(f'Median words per article: {median_words}')
    #print(f'Most common words: {top_words} ({len(word_freqs)} total words)')
    #print(f'Most common bigrams: {top_bigrams} ({len(bigram_freqs)} total bigrams)')
    #print(f'Most common trigrams: {top_trigrams} ({len(trigram_freqs)} total trigrams)')
    #print(f'Most common quadgrams: {top_quadgrams} ({len(quadgram_freqs)} total quadgrams)')
    #print(f'Most common part-of-speech tags: {all_tags}')
    summary = ""
    summary += f'Résultats totaux : {total_results}\n'
    summary += f'Longueur moyenne des articles : {avg_length} characters\n'
    summary += f'Nombre médian de mots par article : {median_words}\n'
    summary += f'Most common words: {top_words} ({len(word_freqs)} total words)\n'
    summary += f'Most common bigrams: {top_bigrams} ({len(bigram_freqs)} total bigrams)\n'
    summary += f'Most common trigrams: {top_trigrams} ({len(trigram_freqs)} total trigrams)\n'
    summary += f'Most common quadgrams: {top_quadgrams} ({len(quadgram_freqs)} total quadgrams)\n'
    #summary += f'Tags les plus courants {all_tags} )\n'
    #summary = '\n'.join(summary)
    #st.markdown(str(summary))
    return summary





#def save_to_file(filename, content):
    #with open(filename, 'w') as f:
        #f.write("\n".join(content))


@st.cache_data(show_spinner=False)
def generate_content(prompt, model="gpt-3.5-turbo", max_tokens=1000, temperature=0.4):
    prompt = truncate_to_token_length(prompt,2500)
    #st.write(prompt)
    #for i in range(3):
        #try:
    gpt_response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "Simuleer een uitzonderlijk getalenteerde journalist en redacteur. Denk stap voor stap na over de volgende instructies en produceer de best mogelijke output."},
            {"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=temperature,
    )
    response = gpt_response['choices'][0]['message']['content'].strip()
    response = response
    return response.strip().split('\n')

        #except:
            #st.write(f"Attempt {i+1} failed, retrying...")
            #time.sleep(3)  # Wait for 3 seconds before next try

    #st.write("OpenAI is currently overloaded, please try again later.")
    #return None

@st.cache_data(show_spinner=False)
def generate_content2(prompt, model="gpt-3.5-turbo", max_tokens=1000, temperature=0.4):
    prompt = truncate_to_token_length(prompt,2500)
    #st.write(prompt)
    #for i in range(3):
        #try:
    gpt_response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "Simuleer een uitzonderlijk getalenteerde journalist en redacteur. Denk stap voor stap na over de volgende instructies en produceer de best mogelijke output. Stuur de resultaten terug in mooi opgemaakte markdown."},
            {"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=temperature,
    )
    response = gpt_response['choices'][0]['message']['content'].strip()
    response = response
    return response

        #except:
            #st.write(f"Attempt {i+1} failed, retrying...")
            #time.sleep(3)  # Wait for 3 seconds before next try

    #st.write("OpenAI is currently overloaded, please try again later.")
    #return None

    
@st.cache_data(show_spinner=False)
def generate_content3(prompt, model="gpt-3.5-turbo", max_tokens=1000, temperature=0.4):
    prompt = truncate_to_token_length(prompt,2500)
    #st.write(prompt)
    #for i in range(3):
        #try:
    gpt_response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "Simuleer een uitzonderlijk getalenteerde onderzoeksjournalist en onderzoeker. Schrijf bij de volgende tekst een korte paragraaf met alleen de belangrijkste feiten en aanknopingspunten die je later kunt gebruiken bij het schrijven van een volledige analyse of artikel."},
            {"role": "user", "content": f"Gebruik de volgende tekst om de uitlezing te geven: {prompt}"}],
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=temperature,
    )
    response = gpt_response['choices'][0]['message']['content'].strip()
    response = response
    return response    
    
    
    
@st.cache_data(show_spinner=False)
def generate_semantic_improvements_guide(prompt,query, model="gpt-3.5-turbo", max_tokens=2000, temperature=0.4):
    prompt = truncate_to_token_length(prompt,1500)
    #for i in range(3):
        #try:
    gpt_response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": """Je bent een expert in semantische SEO. In het bijzonder ben je bovenmenselijk in het nemen van een bepaald NLTK rapport over een bepaald tekstcorpus samengesteld uit de tekst van de gelinkte pagina's die terugkomen bij een google zoekopdracht.
            en het te gebruiken om een uitgebreide set instructies te maken voor een schrijver van een artikel, die kan worden gebruikt om iemand te informeren die een long-form artikel schrijft over een bepaald onderwerp, zodat ze de semantische SEO zoals weergegeven in NLTK-gegevens van het SERP-corpus zo goed mogelijk kunnen behandelen. 
             Geef het resultaat in goed geformatteerde markdown. Het doel van deze gids is om de schrijver te helpen ervoor te zorgen dat de inhoud die ze maken zo volledig mogelijk is voor de semantische SEO met een focus op wat het meest belangrijk is vanuit een semantisch SEO-perspectief."""},
            {"role": "user", "content": f"Semantische SEO-gegevens voor het trefwoord op basis van de inhoud die op de eerste pagina van google staat voor het gegeven trefwoord zoekopdracht van: {query} en zijn gerelateerde semantische gegevens: {prompt}"}],
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=temperature,
    )
    response = gpt_response['choices'][0]['message']['content'].strip()
    st.header("Semantic Improvements Guide")
    st.markdown(response,unsafe_allow_html=True)
    return str(response)

        #except:
            #st.write(f"Attempt {i+1} failed, retrying...")
            #time.sleep(3)  # Wait for 3 seconds before next try

    #st.write("OpenAI is currently overloaded, please try again later.")
    #return None
    
   

@st.cache_data(show_spinner=False)
def generate_outline(topic, model="gpt-3.5-turbo", max_tokens=1500):
    prompt = f"Maak een ongelooflijk grondig artikeloverzicht voor het onderwerp: {topic}. Overweeg alle mogelijke invalshoeken en wees zo grondig mogelijk. Gebruik Romeinse cijfers voor elk onderdeel."
    outline = generate_content(prompt, model=model, max_tokens=max_tokens)
    #save_to_file("outline.txt", outline)
    return outline

@st.cache_data(show_spinner=False)
def improve_outline(outline, semantic_readout, model="gpt-3.5-turbo", max_tokens=1500):
    prompt = f"Als je het volgende artikel schetst, verbeter en breid dit dan zo veel mogelijk uit, rekening houdend met de SEO-zoekwoorden en gegevens die worden verstrekt in onze semantische SEO-uitlezing. Voeg geen sectie toe over semantische SEO zelf, je gebruikt de uitlezing om je beter te informeren bij het maken van het overzicht. Probeer dit zoveel mogelijk op te nemen en uit te breiden. Gebruik Romeinse cijfers voor elke sectie. Het doel is een zo grondig, duidelijk en nuttig mogelijke uitlijning die het onderwerp zo diep mogelijk verkent. Denk stap voor stap na voordat je antwoordt. Houd rekening met de semantische seo-uitlezing die hier wordt gegeven: {semantic_readout} die zou moeten helpen bij het bepalen van de verbeteringen die je kunt aanbrengen, maar denk ook aan extra verbeteringen die niet in deze semantische seo readout zijn opgenomen.  Overzicht om te verbeteren: {outline}."
    improved_outline = generate_content(prompt, model=model, max_tokens=max_tokens)
    #save_to_file("improved_outline.txt", improved_outline)
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
        full_outline = "Gezien het volledige verbeterde overzicht: "
        full_outline += '\n'.join(improved_outline)
        specific_section = "en zich specifiek richten op de volgende sectie: "
        specific_section += section_outline
        prompt =  specific_section + ", schrijf alsjeblieft een grondige paragraaf die de diepte ingaat, die details en bewijzen geeft en die zoveel mogelijk extra waarde toevoegt. Bewaar de hiërarchie die je vindt. Schrijf nooit een conclusie van een sectie, tenzij de sectie zelf een conclusie moet zijn. Tekst van de paragraaf:"
        section = generate_content(prompt, model=model, max_tokens=max_tokens)
        sections.append(section)
        #save_to_file(f"section_{i+1}.txt", section)
    return sections

@st.cache_data(show_spinner=False)
def improve_section(section, i, model="gpt-3.5-turbo", max_tokens=1500):
    prompt = f"Gezien het volgende gedeelte van het artikel: {section}, breng verbeteringen aan in deze sectie. Bewaar de hiërarchie die je vindt. Geef alleen de bijgewerkte sectie, niet de tekst van je aanbeveling, maak alleen de wijzigingen. Geef de bijgewerkte sectie altijd in geldige Markdown. Bijgewerkte sectie met verbeteringen :"
    prompt = str(prompt)
    improved_section = generate_content2(prompt, model=model, max_tokens=max_tokens)
    #st.markdown(improved_section)
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

    #print("Final draft created.\n")
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
    #st.markdown(final_content,unsafe_allow_html=True)
   




def main():
    st.title('Artikelgenerator met semantische SEO-begrip')
    
    st.markdown('''
     Welkom bij de artikelgenerator voor lange artikelen! Deze applicatie maakt gebruik van geavanceerde AI om uitgebreide artikelen te maken op basis van het onderwerp dat je opgeeft. 

    Het genereert niet alleen artikelen, maar het bevat ook een Semantisch SEO begrip. Dit betekent dat het rekening houdt met de semantische context en relevantie van je onderwerp, gebaseerd op de huidige zoekmachineresultaten.

    Voer je onderwerp hieronder in en laat de AI zijn magie doen!
    ''')
   
    topic = st.text_input("Enter topic:", "Kopen in 2050")

    # Get user input for API key
    user_api_key = st.text_input("Voer uw wachtwoord in voor API OpenAI")

    if st.button('Generate Content'):
        if user_api_key:
            openai.api_key = user_api_key
            with st.spinner("Generating content..."):
                final_draft = generate_article(topic)
                #st.markdown(final_draft)
        else:
            st.warning("Please enter your OpenAI API key above.")

if __name__ == "__main__":
    main()






