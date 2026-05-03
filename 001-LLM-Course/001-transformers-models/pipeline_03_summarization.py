import os
import re
from collections import Counter
from transformers import pipeline
from dotenv import load_dotenv


ARTICLE = """Días antes de que dos estudiantes de posgrado de la Universidad del Sur de Florida desaparecieran el mes pasado, un compañero de habitación de uno de ellos supuestamente le hizo una pregunta inusual al chatbot de inteligencia artificial ChatGPT.

"¿Qué pasa si a un humano lo ponen (sic) en una bolsa de basura negra y lo tiran en un contenedor?", preguntó Hisham Abugharbieh el 13 de abril, según una declaración jurada presentada por fiscales de Florida.

ChatGPT respondió que sonaba peligroso, indica el documento, y Abugharbieh hizo entonces otra pregunta: "¿Cómo lo descubrirían?".

Esas supuestas entradas en ChatGPT, incluidas en documentos judiciales que acusan a Abugharbieh de dos cargos de homicidio premeditado, son solo el ejemplo más reciente de investigadores que utilizan historiales de chats con IA como evidencia en investigaciones criminales. Una conversación con ChatGPT también se utilizó en el caso de incendio provocado durante los incendios forestales de Los Ángeles, y una conversación con la IA de Snapchat fue una prueba clave en un juicio por asesinato en Virginia en 2024.

Para los investigadores, estos registros de chat pueden ofrecer información valiosa sobre el estado mental y el posible motivo de un sospechoso.

"Creo que cualquier comunicación con chatbots de IA es como un tesoro para las agencias de seguridad", dijo Ilia Kolochenko, experto en ciberseguridad y abogado en Washington, DC.

Los casos criminales subrayan el creciente uso de los chatbots de IA para obtener consejos personales y la falta de protecciones de privacidad para esas conversaciones. Aunque los chatbots de IA se han convertido rápidamente en una fuente habitual para asesoría legal, diagnósticos médicos y terapia, esas conversaciones no están protegidas legalmente como lo estarían con un abogado, médico o terapeuta con licencia.

El CEO de OpenAI, Sam Altman, ha señalado que esta falta de privacidad es un "gran problema"."""


def load_summarizer():
    load_dotenv(".secrets")
    return pipeline("summarization", model="facebook/bart-large-cnn")


def basic_example(summ):
    result = summ(ARTICLE, max_length=130, min_length=30, do_sample=False, truncation=True)
    print(result[0]["summary_text"])


def abstractive_vs_extractive(summ):
    print("=== ABSTRACTIVE (BART) ===")
    result = summ(ARTICLE, max_length=130, min_length=30, do_sample=False, truncation=True)
    print(result[0]["summary_text"])

    print("\n=== EXTRACTIVE (word frequency) ===")
    print(_extractive_summary(ARTICLE))


def _extractive_summary(text, n=3):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    words = re.findall(r'\b\w+\b', text.lower())
    freq = Counter(words)
    stop = {'que', 'de', 'la', 'el', 'en', 'un', 'y', 'a', 'los', 'se', 'con', 'por', 'su',
            'the', 'of', 'in', 'and', 'to', 'is', 'una', 'o', 'no', 'las', 'es', 'si', 'lo'}
    for w in stop:
        freq.pop(w, None)
    scores = [
        (sum(freq.get(w, 0) for w in re.findall(r'\b\w+\b', s.lower())) / len(s.split()), s)
        for s in sentences
    ]
    return ' '.join(s for _, s in sorted(scores, reverse=True)[:n])


if __name__ == "__main__":
    summarizer = load_summarizer()
    basic_example(summarizer)
    abstractive_vs_extractive(summarizer)
