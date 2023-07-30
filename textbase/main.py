import textbase
from textbase.message import Message
from textbase import models
import os
from typing import List
import random
from transformers import pipeline
import requests
from dotenv import load_dotenv
import os


"""

This file, main.py, includes the implementation of the MentalHealthSupportBot, which is a chatbot designed to offer mental health support to users. The following points summarize its functionalities:

- The MentalHealthSupportBot maintains a `counter` and the `state` of the conversation, which helps to determine the next actions of the chatbot.

- It analyzes the sentiment of the user's input using a sentiment analysis pipeline (uses model distilbert-base-uncased-finetuned-sst-2-english). The analysis output is used to decide the appropriate response.

- Various prompts are used to generate the bot's response. The selection of prompts is based on the sentiment of the user's input and the state of the conversation. For example:
    - Reassuring prompts are used for negative sentiment inputs.
    - Encouraging prompts are used for positive sentiment inputs.
    - Solution-oriented prompts are used for neutral sentiment inputs.
    - Wrap-up prompts are used if the conversation has been ongoing for a significant amount of time.

- It fetches random positive quotes and jokes from external APIs to include in the response when the situation is appropriate.

- The chatbot uses the GPT-3.5 model from OpenAI to generate its responses. 

- It further analyzes the sentiment of its own responses. If the analysis indicates a severe situation, the bot responds with a standard message directing the user to seek professional help.

- The history of prompts used and the sentiment analysis of the messages are logged into a file, 'prompt_history.txt'.

- This script sets the OpenAI API key, which is necessary for making API requests to generate the chatbot's responses.
"""

# Load your OpenAI API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")



# Define prompts
PROMPTS = {
    'guideline': "make sure you dont include any topic which can trigger the user based on conversation history and your understanding of user. Dont make use of negative suggestions",
    'positive_quote': "I am giving some positive quotes, quote any of them to the user, add follow up points,",
    'joke': "I am giving some jokes, quote any of them to the user, add follow up points to make user laugh, ",
    'encouragement': "The user needs encouragement, ",
    'reassurance': "Reassure the user that everything will be fine, ",
    'solution': "Give some solutions for user's problems, ",
    'wrap_up': "The conversation has lasted too long, bid goodbye to the user in friendly manner, "
}

def getPositiveQuotes():
    response = requests.get("https://type.fit/api/quotes")
    data = response.json()
    return str(data[0])

def getJokes():
    limit = 1
    api_url = 'https://api.api-ninjas.com/v1/jokes?limit={}'.format(limit)
    response = requests.get(api_url, headers={'X-Api-Key': 'VZwB1hcpGVNXVdc+tVZ7CA==9aFy9zC05Qh8n1O7'})
    if response.status_code == requests.codes.ok:
        return response.text
    else:
        print("Error:", response.status_code, response.text)


@textbase.chatbot("mental-health-support-bot")
def on_message(message_history: List[Message], state: dict = None):
    if state is None or "counter" not in state:
        state = {"counter": 0, "stage": "greeting"}
    else:
        state["counter"] += 1

    
    # Build the list of message objects
    messages = [{'role': m.role, 'content': m.content} for m in message_history]

    last_user_message = message_history[-1].content if message_history else ""
    
    # Sentiment Analysis
    sentiment_pipeline = pipeline("sentiment-analysis")
    sentiment = sentiment_pipeline(last_user_message)[0]


    # Decide prompt message based on sentiment
    prompt_message = ""

    if state["counter"] % 100 == 0:
        # Time to wrap up
        prompt_message = PROMPTS['wrap_up'] + PROMPTS['guideline']

    if sentiment['label'] == 'NEGATIVE' and sentiment['score']>0.95:
        prompt_message = PROMPTS['reassurance'] + PROMPTS['guideline']
        state["stage"] = "Negative"

    elif sentiment['label'] == 'POSITIVE' and sentiment['score']>0.95:
        prompt_message = PROMPTS['encouragement'] + PROMPTS['guideline']
        x = random.random()
        if(x<=0.25):
            prompt_message += PROMPTS['positive_quote'] + getPositiveQuotes() + PROMPTS['guideline']
        elif(x<=0.5):
            prompt_message += PROMPTS['joke'] + getJokes() + PROMPTS['guideline']
        state["stage"] = "Positive"
    else:
        prompt_message = PROMPTS['solution'] + PROMPTS['guideline']
        state["stage"] = "Solution"

    # Add the prompt message to messages
    messages.append({'role': 'system', 'content': prompt_message})

    # Generate GPT-3.5 Turbo response
    bot_response = models.OpenAI.generate(prompt_message, messages)
    
    bot_emotion = sentiment_pipeline(bot_response)[0]

    with open("prompt_history.txt", "a") as f:
        f.write('User state & Prompt\n'+ state['stage'] + '  ->  ' + prompt_message + "\n" + 'User message sentiment analysis\n' + sentiment['label'] + "/ " + str(sentiment['score']) + "\n" + 'Bot message sentiment analysis\n' + bot_emotion['label'] + "/ " + str(bot_emotion['score']) + "\n\n\n" )

    # Check if the bot response indicates a severe situation
    if (bot_emotion['label']=='NEGATIVE' and bot_emotion['score']>0.99) or 'suicide' in bot_response or 'harm myself' in bot_response or 'self harm' in bot_response:
        bot_response = 'I\'m really sorry that you\'re feeling this way, but I\'m unable to provide the help that you need. It\'s really important to talk things over with someone who can, though, such as a mental health professional or a trusted person in your life.'

    return bot_response, state




models.OpenAI.api_key = openai_api_key