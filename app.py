from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import spacy
import random
import PyPDF2
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import nltk
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer, util
import numpy as np
from collections import defaultdict




# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

app = Flask(__name__)
Bootstrap(app)

# Load English tokenizer, tagger, parser
nlp = spacy.load("en_core_web_sm")

# Load models
model_name = "fares7elsadek/t5-base-finetuned-question-generation"
tokenizer = AutoTokenizer.from_pretrained(model_name)
qg_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
distractor_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def generate_qa(context: str, answer: str, max_len=64):
    """Generate question-answer pair from context"""
    input_text = f"context: {context} answer: {answer} </s>"
    inputs = tokenizer([input_text], return_tensors="pt", truncation=True, padding=True)
    output = qg_model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_len,
        num_beams=5,
        early_stopping=True
    )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    if "question:" in text and "answer:" in text:
        q, a = text.split("answer:")
        return q.replace("question:", "").strip(), a.strip()
    if "?" in text:
        q, a = text.split("?", 1)
        return (q + "?").strip(), a.strip()
    return None, None

def get_distractors(answer: str, context: str, num_distractors=3):

    norm_answer = answer.lower()
    """Generate high-quality distractors using multiple strategies"""
    # Strategy 1: Semantic similarity using sentence transformers
    try:
        # Get similar words from the context
        context_words = [token.text for token in nlp(context) 
                        if token.is_alpha and token.text.lower() != norm_answer]
        
        if context_words:
            # Get embeddings for answer and context words
            answer_embedding = distractor_model.encode([answer], convert_to_tensor=True)
            context_embeddings = distractor_model.encode(context_words, convert_to_tensor=True)
            
            # Calculate cosine similarities
            similarities = util.pytorch_cos_sim(answer_embedding, context_embeddings)[0]
            
            # Get top similar words
            top_indices = similarities.argsort(descending=True)[:num_distractors*2]
            # Collect unique distractors (case-insensitive)
            unique_distractors = []
            seen = set()
            for i in top_indices:
                word = context_words[i]
                norm_word = word.lower()
                if norm_word not in seen and norm_word != norm_answer:
                    unique_distractors.append(word)
                    seen.add(norm_word)
                    if len(unique_distractors) >= num_distractors:
                        break
            
            if len(unique_distractors) >= num_distractors:
                return unique_distractors
    except:
        pass
    
    # Strategy 2: WordNet-based distractors
    key = answer.replace(" ", "_")
    synsets = wn.synsets(key, pos='n')
    if synsets:
        # Get hypernyms -> hyponyms (siblings)
        hyper = synsets[0].hypernyms()
        if hyper:
            hypos = hyper[0].hyponyms()
            distros = [lemma.name().replace("_", " ") for h in hypos for lemma in h.lemmas()]
            #distros = list(set(distros) - {answer})[:num_distractors]
            unique_distros = []
            seen = set()
            for word in distros:
                norm_word = word.lower()
                if norm_word not in seen and norm_word != norm_answer:
                    unique_distros.append(word)
                    seen.add(norm_word)
            distros = unique_distros[:num_distractors]
            if distros:
                return distros
        
        # Get direct synonyms
        synonyms = set()
        for syn in synsets:
            for lemma in syn.lemmas():
                word = lemma.name().replace("_", " ")
                if word.lower() != answer.lower():
                    synonyms.add(word)
        synonyms = list(synonyms)[:num_distractors]
        if synonyms:
            return synonyms
    
    # Strategy 3: Context-based nouns
    doc = nlp(context)
    nouns = [token.text for token in doc if token.pos_ == "NOUN" 
             and token.text != answer and len(token.text) > 3]
    
    # Deduplicate nouns
    unique_nouns = []
    seen = set()
    for noun in nouns:
        norm_noun = noun.lower()
        if norm_noun not in seen:
            unique_nouns.append(noun)
            seen.add(norm_noun)
            
    return random.sample(nouns, min(num_distractors, len(nouns))) if nouns else []

def generate_mcqs(text: str, num_questions=5):
    """Generate high-quality MCQs from text"""
    doc = nlp(text)
    # Filter sentences with at least 8 words
    sentences = [s.text.strip() for s in doc.sents if len(s.text.split()) >= 8]
    
    if not sentences:
        return []
    
    # Extract all nouns for distractor pool
    all_nouns = [token.text for token in doc if token.pos_ == "NOUN" and len(token.text) > 3]
    
    # Precompute sentence embeddings for similarity filtering
    sentence_embeddings = distractor_model.encode(sentences, convert_to_tensor=True)
    
    mcqs = []
    used_sentences = set()
    
    # Process up to requested number of questions
    while len(mcqs) < num_questions and len(used_sentences) < len(sentences):
        # Find the most dissimilar sentence
        if mcqs:
            last_embedding = distractor_model.encode([mcqs[-1][0]], convert_to_tensor=True)
            similarities = util.pytorch_cos_sim(last_embedding, sentence_embeddings)[0]
            idx = similarities.argmin().item()
        else:
            idx = random.randint(0, len(sentences)-1)
            
        if idx in used_sentences:
            # Find next available sentence
            available = [i for i in range(len(sentences)) if i not in used_sentences]
            if not available: 
                break
            idx = random.choice(available)
        
        sent = sentences[idx]
        used_sentences.add(idx)
        
        # Process sentence
        sent_doc = nlp(sent)
        
        # Extract entities (people, orgs, locations, nouns)
        entities = []
        for ent in sent_doc.ents:
            if ent.label_ in ("PERSON", "ORG", "GPE", "LOC") and 3 <= len(ent.text) <= 25:
                entities.append(ent.text)
        
        # Fallback to important nouns
        if not entities:
            nouns = [token.text for token in sent_doc 
                    if token.pos_ == "NOUN" and token.text in all_nouns]
            if nouns:
                entities.append(random.choice(nouns))
        
        if not entities:
            continue
            
        # Use first entity as answer
        answer_ent = entities[0]
        question, answer = generate_qa(sent, answer=answer_ent)
        
        if not question or not answer:
            continue
            
        # Get distractors using multiple strategies
        distractors = get_distractors(answer, text)
        if len(distractors) < 3:
            # Fallback to random nouns from text
            distractors = random.sample(all_nouns, min(3, len(all_nouns))) if all_nouns else []
        
        options = [answer]
        seen_options = {answer.lower()}
        
        for distractor in distractors:
            norm_dist = distractor.lower()
            if norm_dist not in seen_options and norm_dist != answer.lower():
                options.append(distractor)
                seen_options.add(norm_dist)
                if len(options) == 4:  # We need exactly 4 options
                    break
        
        if len(options) < 4:
            # Fill with unique nouns if needed
            for noun in all_nouns:
                norm_noun = noun.lower()
                if norm_noun not in seen_options and norm_noun != answer.lower():
                    options.append(noun)
                    seen_options.add(norm_noun)
                    if len(options) == 4:
                        break
        
        if len(options) < 4:
            continue
            
        random.shuffle(options)
        correct_letter = chr(65 + options.index(answer))  # A, B, C, D
        
        mcqs.append((question, options, correct_letter))
            
    return mcqs

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = ""
        files = request.files.getlist('files[]')
        
        # Process uploaded files
        for file in files:
            if file.filename.endswith('.pdf'):
                text += process_pdf(file)
            elif file.filename.endswith('.txt'):
                text += file.read().decode('utf-8')
        
        # Fallback to text input if no files
        if not text:
            text = request.form.get('text', '')
        
        if not text.strip():
            return render_template('index.html', error="Please provide text or upload a file")
        
        # Get number of questions from form
        num_questions = int(request.form['num_questions'])
        
        # Limit to reasonable number of questions
        num_questions = min(max(num_questions, 1), 20)
        
        mcqs = generate_mcqs(text, num_questions=num_questions)
        
        # Format for template (add index)
        mcqs_with_index = [(i + 1, mcq) for i, mcq in enumerate(mcqs)]
        return render_template('mcqs.html', mcqs=mcqs_with_index)
    
    return render_template('index.html')

def process_pdf(file):
    """Extract text from PDF file"""
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

if __name__ == '__main__':
    app.run(debug=True)