# def map_relational_data(data, mapper):
#     """
#     Dynamically maps relational data based on a predefined schema (mapper).
    
#     :param data: Dictionary containing datasets (tables)
#     :param mapper: Dictionary defining relationships between tables
#     :return: List of merged records
#     """

#     def process_record(record, mapping):
#         """ Recursively merge child data based on FK relationships """
#         merged_record = record.copy()
#         print("MERGED RECORD.....", merged_record)

#         if "child" in mapping:
#             for child_name, child_info in mapping["child"].items():
#                 fk = child_info["fk"]

#                 # Get the dataset for the child (if it exists)
#                 child_dataset = data.get(child_name, [])

#                 # Find matching child record based on FK
#                 related_record = next((item for item in child_dataset if item.get("id") == record.get(fk)), None)
#                 print("RELATED DATA.....", related_record, child_info)
#                 if related_record:
#                     merged_record.update(process_record(related_record, child_info))  # Recursive merge

#         return merged_record

#     # Identify the top-level parent dataset
#     parent_table = list(mapper.keys())[0]
#     parent_mapping = mapper[parent_table]

#     # Get the dataset for the parent table (if available)
#     parent_dataset = data.get(parent_table, [])

#     # Process each record in the parent dataset
#     merged_data = [process_record(record, parent_mapping) for record in parent_dataset]

#     return merged_data


# # 🔹 Example Predefined Mapper (Schema)
# mapper = {
#     "posts":{
#         "child":{
#             "users":{
#                 "fk": "Author"
#             }
#         }
#     },
#     "categories":{
        
#     }
#     "ordersdetails": {
#         "child": {
#             "products": {
#                 "fk": "ProductID",
#                 "child": {
#                     "categories": {
#                         "fk": "CategoryID"
#                     }
#                 }
#             },
#             "orders": {
#                 "fk": "OrderID",
#                 "child": {
#                     "customers": {
#                         "fk": "CustomerID"
#                     }
#                 }
#             }
#         }
#     }
# }

# # 🔹 Example Incoming Data (Some datasets may be missing)
# data = {
#     "ordersdetails": [{"id": 1, "OrderID": 101, "ProductID": 3}],
#     "orders": [{"id": 101, "CustomerID": 5, "total": 500}],
#     "products": [{"id": 3, "name": "Laptop", "CategoryID": 2}],
#     "customers": [{"id": 5, "name": "Rahul"}]
#     # Missing "categories" dataset
# }

# # 🔹 Run Mapping Function
# result = map_relational_data(data, mapper)

# # 🔹 Print Mapped Output
# import json
# print(json.dumps(result, indent=2))


from woocommerce import API

import initial

wcapi = API(
    url="http://localhost:10003",
    consumer_key="ck_6dda33be2f01d1dba75e839107643f820d284804",
    consumer_secret="cs_b6eae529204d1b8bdf34f615a693f3b8bbf048f9",
    version="wc/v3"   #Woocommerce
    # version="wp/v2"  #Wordpress
  )

# FOR PRODUCTS
# r = wcapi.get("products", params={
#     "per_page":100,
#     "page":1,
#     "_fields":"slug,permalink,description,price,sale_price,on_sale,average_rating,categories,tags,stock_status",
#     # "parent":"0", #for fetching only parents
#     "include":"1,2,155"  #for fetching parent variations
# })


# FOR ORDERS
# r = wcapi.get("orders", params={
#     "per_page":100,
#     "page":1,
#     "_fields":"status,currency,date_created,discount_total,shipping_total,total,customer_id,billing,line_items"
#     # "status":"pending,processing,on-hold,completed,cancelled,refunded,failed"
# })

# FOR CART
# r = wcapi.get("orders", params={
#     "per_page":100,
#     "page":1,
#     "_fields":"total,line_items",
#     "status":"checkout-draft"
# })

# FOR CATEGORIES
# r = wcapi.get("products/categories", params={
#     "per_page":100,
#     "page":1,
#     "_fields":"name,description",
# })


# FOR USERS
# wcapi = API(
#     url="http://localhost:10003",
#     consumer_key="vishal",
#     consumer_secret="TghM 5Mw5 Mniv 3ATw goXS DpWU",
#     version="wp/v2"  #Wordpress
#   )
# wcapi.is_ssl = True
# wcapi.query_string_auth = False


# r = wcapi.get("users", params={
#     "per_page":100,
#     "page":1, 
# })

# r = wcapi.get("posts", params={
#     "per_page":1,
#     "page":1
# })

# print("...........", r.json())



# EMBEDDING BASE SIMILARITY CHECKER
# import spacy

# # Load the English NLP model
# nlp = initial.NLP_PROCESSOR

# import spacy
# from sentence_transformers import SentenceTransformer, util
# from sklearn.metrics.pairwise import cosine_similarity

# # Load NLP model for Coreference Resolution
# nlp = spacy.load("en_core_web_sm")

# # Load Sentence Transformer Model for Semantic Similarity
# model = SentenceTransformer("all-MiniLM-L6-v2")

# def ngram_overlap(prev_query, curr_query, n=2):
#     """Calculate N-gram overlap (bigram/trigram similarity)"""
#     prev_ngrams = set(zip(*[prev_query.split()[i:] for i in range(n)]))
#     curr_ngrams = set(zip(*[curr_query.split()[i:] for i in range(n)]))
    
#     intersection = prev_ngrams.intersection(curr_ngrams)
#     union = prev_ngrams.union(curr_ngrams)
    
#     return len(intersection) / len(union) if union else 0

# def calculate_similarity(prev_query, curr_query, cosine_threshold=0.4, ngram_threshold=0.2):
#     """Check if the current query is a follow-up of the previous one."""
    
#     # Cosine similarity
#     prev_embedding = model.encode(prev_query, convert_to_tensor=True)
#     curr_embedding = model.encode(curr_query, convert_to_tensor=True)
#     cosine_sim = util.pytorch_cos_sim(prev_embedding, curr_embedding).item()

#     # N-gram similarity (bigram)
#     bigram_sim = ngram_overlap(prev_query, curr_query, n=2)

#     # Print scores
#     print(f"SIMILARITY SCORE Cosine: {cosine_sim:.4f}")
#     print(f"SIMILARITY SCORE N-Gram (Bigram): {bigram_sim:.4f}")

#     # If either similarity metric is high, it's a follow-up
#     if cosine_sim > cosine_threshold or bigram_sim > ngram_threshold:
#         return True


# def is_followup(prev_query, new_query):
#     """
#     Detect if the new query is a follow-up statement.
#     Combines:
#     1. Semantic similarity (cosine similarity)
#     2. Coreference resolution (pronouns like "it", "that")
#     3. Heuristic rule (keywords like "what about", "how about")
#     """
#     if not prev_query:  # If no previous query, it cannot be a follow-up
#         return False

#     if calculate_similarity(prev_query, new_query):
#         print("calculate_similarity::::::: ", True)
#     print("Regex...........")
#     print(any(new_query for pattern in initial.FOLLOW_UP_PATTERN if pattern.search(new_query)))


# Interactive chatbot loop
# previous_query = None

# print("\n🤖 Chatbot Follow-Up Detector (Type 'exit' to stop)\n")

# while True:
#     previous_query = input("previous: ").strip()
#     user_query = input("current: ").strip()

#     if user_query.lower() == "exit":
#         print("\n👋 Exiting chatbot. Goodbye!\n")
#         break
#     is_followup(previous_query, user_query)



# NLP BASE CHECKER
# def extract_useful_nouns(text):
#     """Extracts useful nouns from a sentence while filtering out generic ones."""
#     doc = nlp(text)
#     useful_nouns = set()

#     # for token in doc:
#         # Check if token is a noun or proper noun and not in the excluded list
#         # if token.pos_ in {"NOUN", "PROPN"}:
            
#         #     useful_nouns.add(token.text)

#     for token in doc:
#       print(f"Token: {token.text}")
#       print(f"  Lemma: {token.lemma_}")
#       print(f"  POS: {token.pos_}")
#       print(f"  Detailed POS Tag: {token.tag_}")
#       print(f"  Dependency: {token.dep_}")
#       print(f"  Named Entity Type: {token.ent_type_}")
#       print(f"  Stopword: {token.is_stop}")
#       print(f"  Is Alpha: {token.is_alpha}")
#       print(f"  Is Numeric: {token.is_digit}")
#       print(f"  Shape: {token.shape_}")
#       print("-" * 40)


# while(True):
    
#     user_query = input("current: ").strip()

#     if user_query.lower() == "exit":
#         print("\n👋 Exiting chatbot. Goodbye!\n")
#         break
#     extract_useful_nouns(user_query)


# LLM BASE CHECKER
import time
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

# Define Groq API Key and Model


def detect_follow_up(prev_query, current_query):
    """Detects if the current query is a follow-up to the previous one."""
    prompt = f"Determine if the second query is a follow-up to the first.\n\nPrevious: {prev_query}\nCurrent: {current_query}\nAnswer with 'yes' or 'no'."
    
    start_time = time.time()
    response = initial.MODELS['small'].invoke([HumanMessage(content=prompt)])
    end_time = time.time()

    answer = response.content.strip().lower()
    is_follow_up = "yes" in answer
    
    return is_follow_up, round(end_time - start_time, 3)

# User input loop
while True:
    prev_query = input("Previous Query: ")
    current_query = input("Current Query: ")
    
    if prev_query.lower() == "exit" or current_query.lower() == "exit":
        break

    is_follow_up, exec_time = detect_follow_up(prev_query, current_query)
    print(f"Follow-up Detected: {is_follow_up}")
    print(f"Execution Time: {exec_time} seconds\n")

