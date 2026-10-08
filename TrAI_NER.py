import streamlit as st
import spacy
import re
import ollama
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from bs4 import BeautifulSoup
from collections import Counter
from wordcloud import WordCloud

def calculate_mattr(tokens, window=100):
    if len(tokens) < window:
        return round(len(set(tokens)) / len(tokens), 4) if tokens else 0
    ttr_sum = sum(len(set(tokens[i:i + window])) / window for i in range(len(tokens) - window + 1))
    return round(ttr_sum / (len(tokens) - window + 1), 4)

def calculate_sttr(tokens, chunk=100):
    if len(tokens) < chunk:
        return round(len(set(tokens)) / len(tokens), 4) if tokens else 0
    chunks = [tokens[i:i + chunk] for i in range(0, len(tokens), chunk) if len(tokens[i:i + chunk]) == chunk]
    if not chunks:
        return 0
    return round(sum(len(set(c)) / chunk for c in chunks) / len(chunks), 4)

def count_matching_sentences(matching_sentences: list) -> dict:
    """
    Counts the number of semantically matching sentences and returns the exact count and list.
    """
    return {
        "count": len(matching_sentences),
        "sentences": matching_sentences
    }

# Define the Ollama tool definition schema
tool_definition = {
    'type': 'function',
    'function': {
        'name': 'count_matching_sentences',
        'description': 'Count the exact number of matching sentences and return their count and list. Call this tool when you need to count, tally, or list sentences matching a specific semantic criteria.',
        'parameters': {
            'type': 'object',
            'properties': {
                'matching_sentences': {
                    'type': 'array',
                    'items': {
                        'type': 'string'
                    },
                    'description': 'The list of ONLY those sentences that match the semantic criteria. Do NOT include non-matching sentences.'
                }
            },
            'required': ['matching_sentences']
        }
    }
}

def submit_query(cat_name):
    input_key = f"q_input_{cat_name}"
    query = st.session_state[input_key].strip()
    if query:
        st.session_state[f"temp_query_{cat_name}"] = query
        st.session_state[input_key] = ""

@st.cache_data(show_spinner=False)
def process_corpus_data(raw_text, lang_model):
    docs_count = len(re.findall(r'<\s*doc\b[^>]*>', raw_text, re.IGNORECASE))
    parts = re.split(r'(<\s*doc\b[^>]*>)', raw_text, flags=re.IGNORECASE)
    texts_with_metadata = []
    current_source = "Unknown Source"
    
    for part in parts:
        if re.match(r'<\s*doc\b', part, re.IGNORECASE):
            tag_soup = BeautifulSoup(part + "</doc>", "html.parser")
            if tag_soup.doc and tag_soup.doc.attrs:
                current_source = ", ".join([f"{k}: {v}" for k, v in tag_soup.doc.attrs.items()])
            else:
                current_source = "Unknown Source"
        else:
            if part.strip():
                clean_text = BeautifulSoup(part, "html.parser").get_text(separator=" ").strip()
                if clean_text:
                    texts_with_metadata.append((clean_text, current_source))
    
    nlp = spacy.load(lang_model, disable=["parser"])
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    nlp.max_length = 2000000
    
    cat_filters = {"Person": ['PER', 'PERSON'], "Organization": ['ORG'], "Location": ['LOC', 'GPE'], "MISC": ['MISC', 'WORK_OF_ART']}
    all_labels = [label for labels in cat_filters.values() for label in labels]
    sents_data = {k: [] for k in ["All"] + list(cat_filters.keys())}
    entities_freq = {k: Counter() for k in cat_filters.keys()}
    
    all_words = []
    total_tokens = 0
    total_ner = 0
    total_sents = 0
    total_lexical = 0
    doc_stats = []

    for doc_spacy, src in nlp.pipe(texts_with_metadata, as_tuples=True, batch_size=10):
        total_tokens += len(doc_spacy)
        ner_in_doc = len(doc_spacy.ents)
        total_ner += ner_in_doc
        sents_list = list(doc_spacy.sents)
        total_sents += len(sents_list)
        
        for t in doc_spacy:
            if t.is_alpha:
                all_words.append(t.text.lower())
                if t.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']:
                    total_lexical += 1
        
        doc_stats.append({
            "source": src,
            "ner_count": ner_in_doc,
            "tokens": len(doc_spacy)
        })

        for sent in sents_list:
            # Highlight all matched entities for "All" category
            all_ents = [ent for ent in sent.ents if ent.label_ in all_labels]
            if all_ents:
                html_text = ""
                last_idx = 0
                for ent in all_ents:
                    start = ent.start_char - sent.start_char
                    end = ent.end_char - sent.start_char
                    html_text += sent.text[last_idx:start] + f"<mark style='background-color: #ffea80; border-radius: 3px; padding: 0 2px; font-weight: bold;'>{sent.text[start:end]}</mark>"
                    last_idx = end
                html_text += sent.text[last_idx:]
                
                sents_data["All"].append({
                    "raw": sent.text,
                    "html": html_text,
                    "source": src
                })

            for cat_name, labels in cat_filters.items():
                cat_ents = [ent for ent in sent.ents if ent.label_ in labels]
                if cat_ents:
                    html_text = ""
                    last_idx = 0
                    for ent in cat_ents:
                        entities_freq[cat_name][ent.text.strip()] += 1
                        start = ent.start_char - sent.start_char
                        end = ent.end_char - sent.start_char
                        html_text += sent.text[last_idx:start] + f"<mark style='background-color: #ffea80; border-radius: 3px; padding: 0 2px; font-weight: bold;'>{sent.text[start:end]}</mark>"
                        last_idx = end
                    html_text += sent.text[last_idx:]
                    
                    sents_data[cat_name].append({
                        "raw": sent.text,
                        "html": html_text,
                        "source": src
                    })

    unique_ents = sum(len(c) for c in entities_freq.values())
    stats = {
        "tokens": total_tokens, "words": len(all_words),
        "ttr": round(len(set(all_words)) / len(all_words), 4) if all_words else 0,
        "mattr": calculate_mattr(all_words, 100),
        "sttr": calculate_sttr(all_words, 100),
        "ltr": round(total_lexical / len(all_words), 4) if all_words else 0,
        "sents": total_sents, "docs": docs_count, "ner": total_ner,
        "e_ttr": round(unique_ents / total_ner, 4) if total_ner > 0 else 0,
        "e_density": round((total_ner / total_tokens) * 1000, 2) if total_tokens > 0 else 0
    }
    
    return sents_data, stats, entities_freq, doc_stats

st.set_page_config(layout="wide")

st.markdown("""
    <style>
    div.stButton > button[kind="primary"] {
        background-color: #007BFF;
        color: white;
        border: none;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #0056b3;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

st.title("TrAI_NER - Corpus Query Engine")

st.sidebar.header("LLM Settings")
model_options = ["qwen2.5:14b", "llama3", "deepseek-r1:8b"]
model_labels = {
    "qwen2.5:14b": "qwen2.5:14b (performance)",
    "llama3": "llama3 (light)",
    "deepseek-r1:8b": "deepseek-r1:8b (reasoning)"
}
selected_llm = st.sidebar.selectbox(
    "Select the model for Pragmatic analysis:",
    model_options,
    format_func=lambda x: model_labels.get(x, x)
)

if 'stats' not in st.session_state:
    st.session_state['stats'] = {}

st.write("Select, switch or compare language model:")
col1, col2, col3 = st.columns(3)

if 'lang_model' not in st.session_state:
    st.session_state['lang_model'] = None

if col1.button("🇮🇹Italiano"): st.session_state['lang_model'] = "it_core_news_lg"
if col2.button("🇬🇧English"): st.session_state['lang_model'] = "en_core_web_lg"
if col3.button("🇪🇸Español"): st.session_state['lang_model'] = "es_core_news_lg"

uploaded_file = st.file_uploader("Upload .txt file", type="txt")

if uploaded_file and st.session_state['lang_model']:
    bytes_data = uploaded_file.read()
    try:
        raw_text = bytes_data.decode("utf-8")
    except UnicodeDecodeError:
        raw_text = bytes_data.decode("windows-1252", errors="replace")
        
    with st.spinner("Analyzing with SpaCy (Cached & Metadata Mapping)..."):
        sents_data, loaded_stats, entities_freq, doc_stats = process_corpus_data(raw_text, st.session_state['lang_model'])
        st.session_state['stats'] = loaded_stats
        st.session_state['doc_stats'] = doc_stats
        st.session_state['entities_freq'] = entities_freq

    st.sidebar.markdown("---")    
    st.sidebar.subheader("NER Metrics")
    st.sidebar.metric("Total Entities", st.session_state['stats'].get("ner", 0))
    st.sidebar.metric("Entity Density (Per 1k tokens)", st.session_state['stats'].get("e_density", 0))
    st.sidebar.metric("Entity TTR", st.session_state['stats'].get("e_ttr", 0))
    st.sidebar.markdown("---")
    st.sidebar.subheader("Corpus Statistics")
    st.sidebar.metric("Tokens", st.session_state['stats'].get("tokens", 0))
    st.sidebar.metric("Words", st.session_state['stats'].get("words", 0))
    st.sidebar.metric("TTR", st.session_state['stats'].get("ttr", 0))
    st.sidebar.metric("MATTR (Window: 100)", st.session_state['stats'].get("mattr", 0))
    st.sidebar.metric("STTR (Chunk: 100)", st.session_state['stats'].get("sttr", 0))
    st.sidebar.metric("LTR (Lexical Density)", st.session_state['stats'].get("ltr", 0))
    st.sidebar.metric("Sentences", st.session_state['stats'].get("sents", 0))
    st.sidebar.metric("Docs (XML)", st.session_state['stats'].get("docs", 0))

    st.write("---")
    st.subheader("NER Quantitative Statistics")
    
    total_toks = st.session_state['stats'].get("tokens", 1)
    norm_base = 100000 if total_toks >= 5000000 else 10000
    
    full_df_rows = []
    df_rows = []
    cat_totals = {}
    cat_diversity = {}
    global_counter = Counter()
    
    for cat_name, counter in entities_freq.items():
        tot_entities = sum(counter.values())
        cat_totals[cat_name] = tot_entities
        cat_diversity[cat_name] = round(len(counter) / tot_entities, 4) if tot_entities > 0 else 0.0
        global_counter.update(counter)
        for ent_text, freq in counter.items():
            full_df_rows.append({
                "Category": cat_name, "Entity": ent_text, "Raw Frequency": freq,
                f"Normalized (per {norm_base})": round((freq / total_toks) * norm_base, 2)
            })
        for ent_text, freq in counter.most_common(10):
            df_rows.append({
                "Category": cat_name, "Entity": ent_text, "Raw Frequency": freq,
                f"Normalized (per {norm_base})": round((freq / total_toks) * norm_base, 2)
            })
    
    if full_df_rows:
        df_full = pd.DataFrame(full_df_rows).sort_values(by="Raw Frequency", ascending=False)
        st.download_button(label="Download Full Dataset as CSV", data=df_full.to_csv(index=False).encode('utf-8'), file_name='ner_full_dataset.csv', mime='text/csv')
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    df_stats = pd.DataFrame(df_rows)
    
    col_table, col_chart = st.columns([2, 1])
    
    with col_table:
        st.write(f"**Top 10 Entities by Category** (Normalization base: {norm_base} tokens)")
        st.dataframe(df_stats, use_container_width=True, hide_index=True)
        
        csv_data = df_stats.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Top 10 Table as CSV", data=csv_data, file_name='ner_top_entities.csv', mime='text/csv')
        
        fig_tbl, ax_tbl = plt.subplots(figsize=(8, len(df_stats)*0.3 + 1))
        ax_tbl.axis('off')
        if not df_stats.empty:
            tbl = ax_tbl.table(cellText=df_stats.values, colLabels=df_stats.columns, loc='center', cellLoc='center')
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            tbl.scale(1.2, 1.5)
        
        buf_tbl = io.BytesIO()
        fig_tbl.savefig(buf_tbl, format="png", bbox_inches="tight", dpi=300)
        buf_tbl.seek(0)
        plt.close(fig_tbl)
        
        st.download_button(label="Download Table as PNG", data=buf_tbl, file_name='ner_table.png', mime='image/png')
        
    with col_chart:
        st.write("**Entities Distribution by Category**")
        df_chart = pd.DataFrame(list(cat_totals.items()), columns=["Category", "Total Found"]).set_index("Category")
        
        st.bar_chart(df_chart)
        
        fig_chart, ax_chart = plt.subplots(figsize=(5, 4))
        ax_chart.bar(df_chart.index, df_chart["Total Found"], color="#4CAF50", edgecolor="black")
        ax_chart.set_ylabel("Absolute Frequency")
        ax_chart.set_title("Entities Distribution")
        plt.xticks(rotation=45)
        
        buf_chart = io.BytesIO()
        fig_chart.savefig(buf_chart, format="png", bbox_inches="tight", dpi=300)
        buf_chart.seek(0)
        plt.close(fig_chart)
        
        st.download_button(label="Download Dist. Chart as PNG", data=buf_chart, file_name='ner_distribution.png', mime='image/png')
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.write("**Category Diversity (Unique / Total)**")
        for cat, div in cat_diversity.items():
            st.write(f"- **{cat}:** {div}")

    st.write("---")
    st.subheader("Top 20 Most Frequent Entities (Global)")
    
    top_20_global = global_counter.most_common(20)
    df_top_global = pd.DataFrame(top_20_global, columns=["Entity", "Raw Frequency"]).set_index("Entity")
    
    st.bar_chart(df_top_global)
    
    fig_top, ax_top = plt.subplots(figsize=(10, 5))
    ax_top.bar(df_top_global.index, df_top_global["Raw Frequency"], color="#FF9800", edgecolor="black")
    ax_top.set_ylabel("Absolute Frequency")
    ax_top.set_title("Top 20 Entities in Corpus")
    plt.xticks(rotation=45, ha='right')
    
    buf_top = io.BytesIO()
    fig_top.savefig(buf_top, format="png", bbox_inches="tight", dpi=300)
    buf_top.seek(0)
    plt.close(fig_top)
    
    if global_counter:
        wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate_from_frequencies(global_counter)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis('off')
        buf_wc = io.BytesIO()
        fig_wc.savefig(buf_wc, format="png", bbox_inches="tight", dpi=300)
        buf_wc.seek(0)
        plt.close(fig_wc)
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        st.download_button(label="Download Top 20 Chart as PNG", data=buf_top, file_name='ner_top20_global.png', mime='image/png')
    with col_btn2:
        if global_counter: st.download_button(label="Download Entities WordCloud", data=buf_wc, file_name='ner_wordcloud.png', mime='image/png')

    st.write("---")
    st.subheader("Document-Level NER Distribution")
    
    df_docs = pd.DataFrame(st.session_state['doc_stats'])
    if not df_docs.empty and 'tokens' in df_docs.columns:
        df_docs['density'] = (df_docs['ner_count'] / df_docs['tokens'].replace(0, 1)) * 1000
        mean_dens = df_docs['density'].mean()
        median_dens = df_docs['density'].median()
        std_dens = df_docs['density'].std()
        
        col_dist_text, col_dist_chart = st.columns([1, 2])
        
        with col_dist_text:
            st.write("**Central Tendencies (Density per 1k tokens)**")
            st.write(f"- **Mean:** {round(mean_dens, 2)}")
            st.write(f"- **Median:** {round(median_dens, 2)}")
            st.write(f"- **Standard Deviation:** {round(std_dens, 2) if pd.notna(std_dens) else 0.0}")
            st.write("Low SD -> More uniform distribution across documents. " \
            "High SD -> More diverse distribution across documents.")
            
        with col_dist_chart:
            import altair as alt
            
            # Create a premium, interactive Altair histogram
            chart = alt.Chart(df_docs).mark_bar(
                color='#8B5CF6',
                cornerRadiusTopLeft=6,
                cornerRadiusTopRight=6
            ).encode(
                x=alt.X("density:Q", bin=alt.Bin(maxbins=15), title="Entity Density (per 1k tokens)"),
                y=alt.Y("count():Q", title="Number of Documents"),
                tooltip=[
                    alt.Tooltip("density:Q", bin=True, title="Density Range (per 1k tokens)"),
                    alt.Tooltip("count():Q", title="Document Count")
                ]
            ).properties(
                height=300
            ).configure_view(
                strokeWidth=0
            ).configure_axis(
                grid=False
            )
            
            st.altair_chart(chart, use_container_width=True)
            
            # Matplotlib version for download structured beautifully to match
            fig_dist, ax_dist = plt.subplots(figsize=(8, 4))
            sns.histplot(df_docs['density'], kde=True, color="#8B5CF6", ax=ax_dist)
            sns.despine()
            ax_dist.set_title("Entity Density Distribution Across Documents")
            ax_dist.set_xlabel("Entity Density (per 1k tokens)")
            ax_dist.set_ylabel("Number of Documents")
            
            buf_dist = io.BytesIO()
            fig_dist.savefig(buf_dist, format="png", bbox_inches="tight", dpi=300)
            buf_dist.seek(0)
            plt.close(fig_dist)
            st.download_button(label="Download Distribution Chart as PNG", data=buf_dist, file_name='ner_density_distribution.png', mime='image/png')

    st.write("---")
    explore_categories = {
        "All": [],
        "Person": ['PER', 'PERSON'],
        "Organization": ['ORG'],
        "Location": ['LOC', 'GPE'],
        "MISC": ['MISC', 'WORK_OF_ART']
    }
    st.subheader("Explore per NER category")
    tabs = st.tabs(list(explore_categories.keys()))
    
    def reset_page(cat):
        st.session_state[f"page_{cat}"] = 0
    
    for cat_name, tab in zip(explore_categories.keys(), tabs):
        if f"page_{cat_name}" not in st.session_state:
            st.session_state[f"page_{cat_name}"] = 0

        with tab:
            current_list = sents_data[cat_name]
            st.write(f"**{len(current_list)}** sentences with **{'any NER' if cat_name == 'All' else cat_name}** tag.")
            
            col_search, col_llm = st.columns(2)
            
            with col_search:
                search_query = st.text_input(f"Filter in {cat_name}:", key=f"s_{cat_name}", on_change=reset_page, args=(cat_name,))
                filtered = [i for i in current_list if search_query.lower() in i["raw"].lower()] if search_query else current_list
                
                total_items = len(filtered)
                items_per_page = 100
                total_pages = (total_items - 1) // items_per_page + 1 if total_items > 0 else 1
                
                if st.session_state[f"page_{cat_name}"] >= total_pages:
                    st.session_state[f"page_{cat_name}"] = 0
                    
                current_page = st.session_state[f"page_{cat_name}"]
                start_idx = current_page * items_per_page
                end_idx = start_idx + items_per_page
                
                if total_items > 0:
                    st.write(f"Showing results {start_idx + 1} to {min(end_idx, total_items)} of {total_items}:")
                    for i, item in enumerate(filtered[start_idx:end_idx], start_idx + 1):
                        st.markdown(f"**{i}.** {item['html']} *(Source: {item['source']})*", unsafe_allow_html=True)
                else:
                    st.write("No results found.")
                
                col_prev, col_page, col_next = st.columns([1, 2, 1])
                with col_prev:
                    if st.button("Previous 100", key=f"prev_{cat_name}", disabled=(current_page == 0)):
                        st.session_state[f"page_{cat_name}"] -= 1
                        st.rerun()
                with col_page:
                    st.markdown(f"<div style='text-align: center; padding-top: 5px;'>Page {current_page + 1} of {total_pages}</div>", unsafe_allow_html=True)
                with col_next:
                    if st.button("Next 100", key=f"next_{cat_name}", disabled=(current_page >= total_pages - 1)):
                        st.session_state[f"page_{cat_name}"] += 1
                        st.rerun()
            
            with col_llm:
                st.markdown(f"### Pragmatic Analysis ({selected_llm})")
                
                # Initialize chat history for this tab
                chat_key = f"chat_history_{cat_name}"
                if chat_key not in st.session_state:
                    st.session_state[chat_key] = []
                
                # Automatically reset conversation if text filter changes
                filter_key = f"last_filter_{cat_name}"
                current_filter_hash = f"{search_query}_{len(filtered)}"
                if filter_key not in st.session_state or st.session_state[filter_key] != current_filter_hash:
                    st.session_state[chat_key] = []
                    st.session_state[filter_key] = current_filter_hash

                # Visual clear button and title spacing
                col_title, col_clear = st.columns([3, 1])
                with col_clear:
                    if st.session_state[chat_key]:
                        if st.button("🗑️ Clear", key=f"clear_{cat_name}", use_container_width=True):
                            st.session_state[chat_key] = []
                            st.rerun()
                
                # Chat message container
                chat_container = st.container()
                
                with chat_container:
                    for msg in st.session_state[chat_key]:
                        # Skip showing the assistant's tool-call request bubble if content is empty
                        if msg["role"] == "assistant" and msg.get("tool_calls") and not msg.get("content"):
                            continue
                        
                        if msg["role"] == "tool":
                            try:
                                import json
                                tool_res = json.loads(msg["content"])
                                count = tool_res.get("count", 0)
                                st.info(f"⚙️ **Tool Use ({msg.get('name')})**: Conteggio esatto calcolato via Python = **{count}** frasi corrispondenti.")
                            except Exception:
                                st.info(f"⚙️ **Tool Use ({msg.get('name')})** completato.")
                            continue
                            
                        with st.chat_message(msg["role"]):
                            if msg.get("think"):
                                # Keep the thinking steps collapsed in history to save space
                                with st.expander("💭 View Reasoning Steps (Chain of Thought)", expanded=False):
                                    st.markdown(f"<div style='color: #6B7280; font-style: italic; border-left: 2px solid #3B82F6; padding-left: 10px; background-color: #F9FAFB; padding-top: 8px; padding-bottom: 8px; border-radius: 4px; white-space: pre-wrap;'>{msg['think']}</div>", unsafe_allow_html=True)
                            st.markdown(msg["content"])

                # Sleek, form-less text input with immediate enter key submission (no gray form lines)
                st.text_input(
                    "Ask a question about these entities:",
                    placeholder="Type here and press Enter to send...",
                    key=f"q_input_{cat_name}",
                    on_change=submit_query,
                    args=(cat_name,),
                    label_visibility="collapsed"
                )

                # Check if callback submitted a query
                temp_query_key = f"temp_query_{cat_name}"
                if temp_query_key in st.session_state and st.session_state[temp_query_key]:
                    llm_query = st.session_state[temp_query_key]
                    # Consume the query immediately
                    del st.session_state[temp_query_key]
                    
                    if filtered:
                        is_first = (len(st.session_state[chat_key]) == 0)
                        
                        if is_first:
                            context = "\n".join([f"[{i+1}][Source: {item['source']}] {item['raw']}" for i, item in enumerate(filtered[:150])])
                            llm_prompt = f"""You are a computational linguistics expert analyzing a corpus of text. Based on the following sentences tagged with {'NER' if cat_name == 'All' else cat_name} entities, answer the question below.:
{context}

Question: {llm_query}

IMPERATIVE RULES:
1. Answer based EXCLUSIVELY on the sentences provided above.
2. It is strictly forbidden to use your external knowledge or invent sources not present in the text.
3. Always cite the exact source reported between square brackets for every sentence you mention.
4. Reply in the same language as the question.

TOOL USE FOR COUNTING/LISTING (If counting/tallying/listing is requested):
- If the user asks to count, find, or list sentences matching a specific meaning, you must identify those sentences.
- Then, you MUST call the tool 'count_matching_sentences' passing ONLY the filtered matching sentences as arguments. Do not attempt to count them yourself, use the tool.
"""
                        else:
                            llm_prompt = f"""Remember the context sentences and rules provided earlier.
Answer the following question based EXCLUSIVELY on them. Cite exact source labels (e.g. [Source: ...]) for sentences you mention.

Question: {llm_query}"""

                        # Append user prompt and trigger rerun to display user input immediately
                        st.session_state[chat_key].append({"role": "user", "content": llm_query, "prompt": llm_prompt})
                        st.rerun()
                    elif not filtered:
                        st.warning("No sentences found with this filter!")

                # Live assistant response generation trigger
                if st.session_state[chat_key] and st.session_state[chat_key][-1]["role"] == "user":
                    user_msg = st.session_state[chat_key][-1]
                    
                    with chat_container:
                        with st.chat_message("assistant"):
                            try:
                                response_placeholder = st.empty()
                                raw_content = ""
                                
                                # Format all messages for Ollama API
                                ollama_messages = []
                                for m in st.session_state[chat_key][:-1]:
                                    msg_dict = {
                                        "role": m["role"],
                                        "content": m.get("prompt", m["content"])
                                    }
                                    if "tool_calls" in m:
                                        msg_dict["tool_calls"] = m["tool_calls"]
                                    if "name" in m:
                                        msg_dict["name"] = m["name"]
                                    ollama_messages.append(msg_dict)
                                
                                last_msg_dict = {
                                    "role": user_msg["role"],
                                    "content": user_msg.get("prompt", user_msg["content"])
                                }
                                if "tool_calls" in user_msg:
                                    last_msg_dict["tool_calls"] = user_msg["tool_calls"]
                                if "name" in user_msg:
                                    last_msg_dict["name"] = user_msg["name"]
                                ollama_messages.append(last_msg_dict)
                                
                                # Conditionally pass the tool ONLY if using the target model qwen2.5:14b
                                tools_to_pass = [tool_definition] if selected_llm == "qwen2.5:14b" else None
                                
                                # Step 1: Call Ollama without streaming to check for tool calls
                                response = ollama.chat(model=selected_llm, messages=ollama_messages, tools=tools_to_pass)
                                
                                if response.message.tool_calls:
                                    # Executing the tools requested by the model
                                    tool_calls_to_append = []
                                    tool_results_to_append = []
                                    
                                    for tc in response.message.tool_calls:
                                        tc_name = tc.function.name
                                        tc_args = tc.function.arguments
                                        if isinstance(tc_args, str):
                                            import json
                                            try:
                                                tc_args = json.loads(tc_args)
                                            except json.JSONDecodeError:
                                                pass
                                        
                                        if tc_name == 'count_matching_sentences':
                                            matching_sents = tc_args.get('matching_sentences', [])
                                            res = count_matching_sentences(matching_sents)
                                            
                                            tool_calls_to_append.append({
                                                "function": {
                                                    "name": tc_name,
                                                    "arguments": tc_args
                                                }
                                            })
                                            
                                            tool_results_to_append.append({
                                                "role": "tool",
                                                "content": json.dumps(res),
                                                "name": tc_name
                                            })
                                    
                                    # Save tool calls and results to the session state chat history
                                    st.session_state[chat_key].append({
                                        "role": "assistant",
                                        "content": response.message.content or "",
                                        "tool_calls": tool_calls_to_append
                                    })
                                    
                                    for tool_res in tool_results_to_append:
                                        st.session_state[chat_key].append(tool_res)
                                        
                                    # Format messages list again including tool call and tool result
                                    ollama_messages = []
                                    for m in st.session_state[chat_key]:
                                        msg_dict = {
                                            "role": m["role"],
                                            "content": m.get("prompt", m["content"])
                                        }
                                        if "tool_calls" in m:
                                            msg_dict["tool_calls"] = m["tool_calls"]
                                        if "name" in m:
                                            msg_dict["name"] = m["name"]
                                        ollama_messages.append(msg_dict)
                                    
                                    # Step 2: Stream the final answer based on the tool results
                                    stream = ollama.chat(model=selected_llm, messages=ollama_messages, stream=True)
                                    
                                    for chunk in stream:
                                        raw_content += chunk['message']['content']
                                        
                                        # Update UI dynamically during streaming
                                        if "<think>" in raw_content:
                                            if "</think>" in raw_content:
                                                parts = raw_content.split("</think>", 1)
                                                think_process = parts[0].replace("<think>", "").strip()
                                                final_answer = parts[1].strip()
                                                
                                                with response_placeholder.container():
                                                    with st.expander("💭 View Reasoning Steps (Chain of Thought)", expanded=True):
                                                        st.markdown(f"<div style='color: #6B7280; font-style: italic; border-left: 2px solid #3B82F6; padding-left: 10px; background-color: #F9FAFB; padding-top: 8px; padding-bottom: 8px; border-radius: 4px; white-space: pre-wrap;'>{think_process}</div>", unsafe_allow_html=True)
                                                    if final_answer:
                                                        st.markdown(final_answer + "▌")
                                            else:
                                                think_process = raw_content.replace("<think>", "").strip()
                                                with response_placeholder.container():
                                                    with st.expander("💭 Thinking (Chain of Thought)...", expanded=True):
                                                        st.markdown(f"<div style='color: #6B7280; font-style: italic; border-left: 2px solid #3B82F6; padding-left: 10px; background-color: #F9FAFB; padding-top: 8px; padding-bottom: 8px; border-radius: 4px; white-space: pre-wrap;'>{think_process}▌</div>", unsafe_allow_html=True)
                                        else:
                                            response_placeholder.markdown(raw_content + "▌")
                                            
                                    # Save final streaming response
                                    if "<think>" in raw_content and "</think>" in raw_content:
                                        parts = raw_content.split("</think>", 1)
                                        think_process = parts[0].replace("<think>", "").strip()
                                        final_answer = parts[1].strip()
                                        
                                        with response_placeholder.container():
                                            with st.expander("💭 View Reasoning Steps (Chain of Thought)", expanded=False):
                                                st.markdown(f"<div style='color: #6B7280; font-style: italic; border-left: 2px solid #3B82F6; padding-left: 10px; background-color: #F9FAFB; padding-top: 8px; padding-bottom: 8px; border-radius: 4px; white-space: pre-wrap;'>{think_process}</div>", unsafe_allow_html=True)
                                            st.markdown(final_answer)
                                            
                                        st.session_state[chat_key].append({
                                            "role": "assistant",
                                            "content": final_answer,
                                            "think": think_process
                                        })
                                    else:
                                        response_placeholder.markdown(raw_content)
                                        st.session_state[chat_key].append({
                                            "role": "assistant",
                                            "content": raw_content
                                        })
                                else:
                                    # No tool calls: just process first_response content directly
                                    raw_content = response.message.content or ""
                                    if "<think>" in raw_content and "</think>" in raw_content:
                                        parts = raw_content.split("</think>", 1)
                                        think_process = parts[0].replace("<think>", "").strip()
                                        final_answer = parts[1].strip()
                                        
                                        with response_placeholder.container():
                                            with st.expander("💭 View Reasoning Steps (Chain of Thought)", expanded=False):
                                                st.markdown(f"<div style='color: #6B7280; font-style: italic; border-left: 2px solid #3B82F6; padding-left: 10px; background-color: #F9FAFB; padding-top: 8px; padding-bottom: 8px; border-radius: 4px; white-space: pre-wrap;'>{think_process}</div>", unsafe_allow_html=True)
                                            st.markdown(final_answer)
                                            
                                        st.session_state[chat_key].append({
                                            "role": "assistant",
                                            "content": final_answer,
                                            "think": think_process
                                        })
                                    else:
                                        response_placeholder.markdown(raw_content)
                                        st.session_state[chat_key].append({
                                            "role": "assistant",
                                            "content": raw_content
                                        })
                                
                                # Rerun to solidify state and avoid double submit bugs
                                st.rerun()
                                    
                            except ollama.ResponseError:
                                st.error(f"Model '{selected_llm}' not found. Open the terminal and run: ollama pull {selected_llm}")