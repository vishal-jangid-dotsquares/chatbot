
# # from woocommerce import API

# # import initial

# # wcapi = API(
# #     url="https://sident.24livehost.com",
# #     consumer_key="ck_4c25c31650af6f4217a3d8169fdda5c5994472d4",
# #     consumer_secret="cs_6cbadd241666701167f3c58329b636d32d753d8e",
# #     version="wc/v3"   #Woocommerce
# #     # version="wp/v2"  #Wordpress
# #   )

# # FOR PRODUCTS
# # r = wcapi.get("products", params={
# #     "per_page":1,
# #     "page":1,
# #     "_fields":"slug,permalink,description,price,sale_price,on_sale,average_rating,categories,tags,stock_status",
# #     "parent":"0", #for fetching only parents
# #     # "include":"1,2,155"  #for fetching parent variations
# # })


# # FOR ORDERS
# # r = wcapi.get("orders", params={
# #     "per_page":1,
# #     "page":1,
# #     "_fields":"status,currency,date_created,discount_total,shipping_total,total,customer_id,billing,line_items"
# #     # "status":"pending,processing,on-hold,completed,cancelled,refunded,failed"
# # })

# # FOR CART
# # r = wcapi.get("orders", params={
# #     "per_page":100,
# #     "page":1,
# #     "_fields":"status, total,line_items",
# #     # "status":"checkout-draft"
# # })

# # FOR CATEGORIES
# # r = wcapi.get("products/categories", params={
# #     "per_page":100,
# #     "page":1,
# #     "_fields":"name,description",
# # })


# # FOR USERS
# # wcapi = API(
# #     url="https://sident.24livehost.com",
# #     consumer_key="vishal",
# #     consumer_secret="TghM 5Mw5 Mniv 3ATw goXS DpWU",
# #     version="wp/v2"  #Wordpress
# #   )
# # wcapi.is_ssl = True
# # wcapi.query_string_auth = False


# # r = wcapi.get("pages", params={
# #     "per_page":100,
# #     "page":1, 
# # })

# # r = wcapi.get("users", params={
# #     "per_page":100,
# #     "page":1, 
# # })

# # r = wcapi.get("posts", params={
# #     "per_page":1,
# #     "page":1
# # })

# # print("...........", r.json())

# from langchain_core.runnables import RunnableLambda, RunnableSequence
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_experimental.text_splitter import SemanticChunker
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.preprocessing import normalize

# import numpy as np

# import initial

# text = """
# Artificial Intelligence, or AI, is a field of computer science focused on creating systems that can perform tasks that normally require human intelligence. These tasks include recognizing speech, understanding language, identifying images, and making decisions. AI is being used in many industries, from healthcare to banking to transportation. In healthcare, AI can help doctors detect diseases early by analyzing medical images or patient history. In agriculture, AI can monitor crops and suggest the best time for harvest. Businesses use AI to improve customer service with chatbots and to analyze large amounts of data quickly. However, there are concerns about AI replacing human jobs, especially in areas like manufacturing or customer service. As AI continues to grow, it's important to use it responsibly and ensure it benefits everyone, not just a few companies or countries.Terrorism is the use of violence or threats by groups or individuals to create fear and achieve political, religious, or ideological goals. It has affected countries all over the world and continues to be a serious threat to global peace and security. Terrorist attacks often target innocent people in public places like airports, markets, or schools. These acts of violence cause death, injury, and emotional trauma for victims and communities. Governments around the world are working together to prevent terrorism by improving security, monitoring suspicious activity, and sharing intelligence. Technology is also being used, such as surveillance cameras and online monitoring tools. But terrorism cannot be solved by force alone. To truly address it, we must also look at the root causes, like poverty, lack of education, and political instability, which sometimes push people toward extremist ideas.Poverty is a condition where people lack enough money to meet their basic needs, such as food, shelter, healthcare, and education. It affects millions of people across the globe, especially in developing countries. In many poor communities, children are malnourished, schools are underfunded, and clean water is hard to find. Poverty also leads to poor health, low education levels, and limited job opportunities. Governments and charities try to reduce poverty through welfare programs, free education, and food distribution. International organizations like the United Nations also run programs to fight global poverty. Still, solving poverty is a complex problem. It requires not only money but also long-term strategies like improving infrastructure, creating jobs, and empowering people with education and skills training. Without addressing poverty, it becomes harder for countries to develop and grow fairly.The education system is the structure through which people learn knowledge, skills, and values. It starts from early childhood education and goes all the way through college and beyond. A strong education system is important for the success of a country. It helps people develop critical thinking, creativity, and the ability to contribute to society. In some countries, students have access to good schools, trained teachers, and modern technology. In others, schools may lack books, electricity, or even proper classrooms. Unequal access to education leads to inequality in society. Some reforms that can help improve education include training teachers better, building more schools in rural areas, and using digital learning tools. Education also helps reduce problems like unemployment and crime. By investing in education, a country invests in its future. Every child deserves the chance to go to school and dream big.The Avengers are a group of fictional superheroes created by Marvel Comics. They became famous through movies produced by Marvel Studios, starting with "The Avengers" in 2012. The team includes characters like Iron Man, Captain America, Thor, Hulk, Black Widow, and Hawkeye. These heroes come from different backgrounds and have unique powers or skills, but they work together to fight villains and save the world. The movies are loved for their action, humor, and emotional stories. They’ve inspired millions of fans and have had a huge impact on pop culture. Beyond entertainment, the Avengers also teach lessons about teamwork, courage, and doing what’s right, even when it’s difficult. While the characters are fictional, their values and stories resonate with real people. Fans around the world collect merchandise, attend conventions, and share their love for these superheroes online. The success of the Avengers has also influenced other movie franchises and led to the creation of a larger Marvel Cinematic Universe.
# """

# # text = """
# # Diabetes is a chronic disease that affects how your body turns food into energy. 
# # There are two main types: type 1 and type 2. Symptoms include excessive thirst, frequent urination, and fatigue. 
# # If left untreated, it can lead to complications such as heart disease and nerve damage. 
# # Proper management includes monitoring blood sugar and maintaining a healthy lifestyle.
# # In the modern world, artificial intelligence (AI) is changing everything around us. From voice assistants on our phones to self-driving cars, AI is becoming part of our daily lives.
# # Terrorism continues to threaten peace and safety in many countries. Innocent people lose their lives, and communities are torn apart by violence and fear.
# # Poverty is another problem that remains a huge burden across the world. In some countries, children don’t have enough food, clean water, or a safe place to sleep.
# # """


# # text = """
# # In the modern world, artificial intelligence (AI) is changing everything around us. From voice assistants on our phones to self-driving cars, AI is becoming part of our daily lives. Some people believe AI can solve many of humanity's biggest problems, including poverty and access to education. For example, AI can help identify students who are struggling in school and recommend better ways to teach them. It can also be used to create personalized lessons so that every student learns at their own pace. However, while AI brings hope, the world still struggles with other serious issues. Terrorism continues to threaten peace and safety in many countries. Innocent people lose their lives, and communities are torn apart by violence and fear. Governments and security agencies use AI tools to track and stop terrorist activities, but it’s a never-ending challenge because new threats appear in different forms.
# # Poverty is another problem that remains a huge burden across the world. In some countries, children don’t have enough food, clean water, or a safe place to sleep. Many kids drop out of school because they need to work and support their families. Without proper education, the cycle of poverty continues, and generations remain stuck without a way out. This is where the education system becomes very important. A good education system helps people grow and improves their chances of getting jobs, making money, and living better lives. But in many places, schools lack basic facilities, qualified teachers, and even electricity. AI could play a role here too, by helping poor countries deliver digital education through mobile devices, even in remote villages.
# # While all of this is happening in the real world, people often escape reality by watching movies and dreaming about superheroes. The Avengers, for example, are a team of heroes who protect the Earth from powerful enemies. Kids and adults alike enjoy their adventures because they represent courage, teamwork, and hope. Imagine if Iron Man's technology could be used to fight real-world problems. His suit is powered by an intelligent system similar to AI. What if we could use that kind of intelligence to build machines that could detect and stop crime before it happens? Or design robots that can build homes, clean polluted rivers, or distribute food to people who need it most?
# # It’s fun to think about, but real change needs real effort. We can’t wait for superheroes to save us. We must be the heroes of our own story. That means building stronger education systems, using AI responsibly, fighting terrorism with better strategies, and solving poverty with smart policies and compassion. AI must be developed in a way that is fair and safe for everyone. Many experts worry that AI could one day get out of control if not managed properly. For example, if companies focus only on profits, they might build systems that replace human workers, leading to more unemployment, especially in poor regions. That would make the poverty problem worse, not better.
# # On the other hand, AI also opens new job opportunities. It creates new industries, such as data science, robotics, and machine learning engineering. But the sad truth is, most poor people don’t have access to these opportunities. Why? Because they never had a chance to learn about technology in the first place. This is why education is so powerful. It’s the bridge between poverty and a better future. In some schools around the world, students are now learning about coding, building robots, and even training small AI models. These are small steps, but they show what is possible.
# # If we also look at the connection between education and terrorism, we find something interesting. Many terrorists come from backgrounds where education is either missing or controlled in a negative way. Without proper understanding of the world, some people fall into hatred and violence. A strong education system teaches critical thinking, empathy, and the value of peace. It helps people make better decisions and reject extremism. So, by fixing the education system, we also help reduce terrorism in the long run.
# # Now, back to the Avengers. Why are they so loved? Maybe because each of them represents something we admire. Captain America stands for honesty and justice. Black Panther represents leadership and responsibility. Hulk shows that even great power needs control. And Iron Man shows the power of innovation. All of these are values we need in real life too. Imagine a world where leaders had the courage of Captain America, the intelligence of Tony Stark, and the compassion of Black Widow. We wouldn’t need to worry about AI going wrong or terrorism spreading fear. We’d be ready to face problems together, like a real-life superhero team.
# # In conclusion, the world is a mix of good and bad, simple and complex. We have tools like AI that can be used for great things, but we must guide them carefully. Poverty and terrorism are deep-rooted problems that won’t go away overnight. However, with the right mix of education, technology, creativity, and teamwork, we can move closer to a better world. We don’t need superpowers to do that. We just need the will to act, learn, and help each other. Just like in the movies, the biggest heroes are not the ones with powers, but the ones who never give up, even when the world seems impossible to fix.
# # """
# # # --------------------
# # CONFIG
# # --------------------
# embedding_fn = initial.EMBEDDING_FUNCTION

# # --------------------
# # STEP 1: Atomic Unit Splitter (Sentences)
# # --------------------
# chunks = RecursiveCharacterTextSplitter(
#                   separators=["\n\n", "\n", fr"(?<=[.?!])\s+"],                                   
#                   keep_separator=False, is_separator_regex=True,
#                   chunk_size=100, chunk_overlap=0)
# # chunks = SemanticChunker(initial.EMBEDDING_FUNCTION, breakpoint_threshold_type="percentile")
# # print(chunks.split_text(text))

# split_sentences = RunnableLambda(lambda text: chunks.split_text(text))
# # split_sentences = RunnableLambda(lambda text: str(text).split("."))

# # --------------------
# # STEP 2: Embed Each Sentence
# # --------------------
# embed_sentences = RunnableLambda(lambda sentences: embedding_fn.embed_documents(sentences))

# # --------------------
# # STEP 3: Semantic Chunking (Merging similar sentences)
# # --------------------

# def merge_semantic_chunks(sentences, embeddings, threshold=0.35, debug=False):
#     """
#     Merge semantically similar sentences into coherent chunks.
    
#     Args:
#         sentences (List[str]): List of input sentences.
#         embeddings (List[List[float]]): Sentence-level embeddings.
#         threshold (float): Cosine similarity threshold for merging.
#         debug (bool): If True, prints debug info.
    
#     Returns:
#         List[str]: List of semantically merged chunks.
#     """
#     if not sentences or not embeddings:
#         return []
#     print("length...", len(sentences))
#     # Normalize all embeddings for stable cosine similarity
#     embeddings = normalize(np.array(embeddings))

#     chunks = []
#     current_chunk = [sentences[0]]
#     current_vectors = [embeddings[0]]

#     if debug:
#         print(f"🔎 Total Sentences: {len(sentences)}")

#     for i in range(1, len(sentences)):
#         avg_vec = np.mean(current_vectors, axis=0).reshape(1, -1)
#         similarity = cosine_similarity(avg_vec, embeddings[i].reshape(1, -1))[0][0]

#         if debug:
#             print(f"→ Similarity with sentence {i}: {similarity:.4f}")

#         if similarity >= threshold:
#             current_chunk.append(sentences[i])
#             current_vectors.append(embeddings[i])
#         else:
#             chunks.append(" ".join(current_chunk))
#             current_chunk = [sentences[i]]
#             current_vectors = [embeddings[i]]

#     # Append the last chunk
#     if current_chunk:
#         chunks.append(" ".join(current_chunk))

#     return chunks

# semantic_chunker = RunnableLambda(
#     lambda input_data: merge_semantic_chunks(input_data["sentences"], input_data["embeddings"])
# )

# # --------------------
# # STEP 4: Final Pipeline
# # --------------------
# semantic_pipeline: RunnableSequence = (
#     split_sentences
#     | (lambda sents: {"sentences": sents, "embeddings": embedding_fn.embed_documents(sents)})
#     | semantic_chunker
# )

# # --------------------
# # Run Pipeline
# # --------------------

# semantic_chunks = semantic_pipeline.invoke(text)

# # --------------------
# # Print Final Semantic Chunks
# # --------------------
# print("📌 Final Semantic Chunks:\n")
# for idx, chunk in enumerate(semantic_chunks, 1):
#     print(f"🔹 Chunk {idx}:\n{chunk}\n")


# # from langchain_experimental.text_splitter import SemanticChunker
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain.text_splitter import RecursiveCharacterTextSplitter

# # import initial

# text = """
# In the modern world, artificial intelligence (AI) is changing everything around us. From voice assistants on our phones to self-driving cars, AI is becoming part of our daily lives. Some people believe AI can solve many of humanity's biggest problems, including poverty and access to education. For example, AI can help identify students who are struggling in school and recommend better ways to teach them. It can also be used to create personalized lessons so that every student learns at their own pace. However, while AI brings hope, the world still struggles with other serious issues. Terrorism continues to threaten peace and safety in many countries. Innocent people lose their lives, and communities are torn apart by violence and fear. Governments and security agencies use AI tools to track and stop terrorist activities, but it’s a never-ending challenge because new threats appear in different forms.
# Poverty is another problem that remains a huge burden across the world. In some countries, children don’t have enough food, clean water, or a safe place to sleep. Many kids drop out of school because they need to work and support their families. Without proper education, the cycle of poverty continues, and generations remain stuck without a way out. This is where the education system becomes very important. A good education system helps people grow and improves their chances of getting jobs, making money, and living better lives. But in many places, schools lack basic facilities, qualified teachers, and even electricity. AI could play a role here too, by helping poor countries deliver digital education through mobile devices, even in remote villages.
# While all of this is happening in the real world, people often escape reality by watching movies and dreaming about superheroes. The Avengers, for example, are a team of heroes who protect the Earth from powerful enemies. Kids and adults alike enjoy their adventures because they represent courage, teamwork, and hope. Imagine if Iron Man's technology could be used to fight real-world problems. His suit is powered by an intelligent system similar to AI. What if we could use that kind of intelligence to build machines that could detect and stop crime before it happens? Or design robots that can build homes, clean polluted rivers, or distribute food to people who need it most?
# It’s fun to think about, but real change needs real effort. We can’t wait for superheroes to save us. We must be the heroes of our own story. That means building stronger education systems, using AI responsibly, fighting terrorism with better strategies, and solving poverty with smart policies and compassion. AI must be developed in a way that is fair and safe for everyone. Many experts worry that AI could one day get out of control if not managed properly. For example, if companies focus only on profits, they might build systems that replace human workers, leading to more unemployment, especially in poor regions. That would make the poverty problem worse, not better.
# On the other hand, AI also opens new job opportunities. It creates new industries, such as data science, robotics, and machine learning engineering. But the sad truth is, most poor people don’t have access to these opportunities. Why? Because they never had a chance to learn about technology in the first place. This is why education is so powerful. It’s the bridge between poverty and a better future. In some schools around the world, students are now learning about coding, building robots, and even training small AI models. These are small steps, but they show what is possible.
# If we also look at the connection between education and terrorism, we find something interesting. Many terrorists come from backgrounds where education is either missing or controlled in a negative way. Without proper understanding of the world, some people fall into hatred and violence. A strong education system teaches critical thinking, empathy, and the value of peace. It helps people make better decisions and reject extremism. So, by fixing the education system, we also help reduce terrorism in the long run.
# Now, back to the Avengers. Why are they so loved? Maybe because each of them represents something we admire. Captain America stands for honesty and justice. Black Panther represents leadership and responsibility. Hulk shows that even great power needs control. And Iron Man shows the power of innovation. All of these are values we need in real life too. Imagine a world where leaders had the courage of Captain America, the intelligence of Tony Stark, and the compassion of Black Widow. We wouldn’t need to worry about AI going wrong or terrorism spreading fear. We’d be ready to face problems together, like a real-life superhero team.
# In conclusion, the world is a mix of good and bad, simple and complex. We have tools like AI that can be used for great things, but we must guide them carefully. Poverty and terrorism are deep-rooted problems that won’t go away overnight. However, with the right mix of education, technology, creativity, and teamwork, we can move closer to a better world. We don’t need superpowers to do that. We just need the will to act, learn, and help each other. Just like in the movies, the biggest heroes are not the ones with powers, but the ones who never give up, even when the world seems impossible to fix.
# """
# text = """
# Diabetes is a chronic disease that affects how your body turns food into energy. 
# There are two main types: type 1 and type 2. Symptoms include excessive thirst, frequent urination, and fatigue. 
# If left untreated, it can lead to complications such as heart disease and nerve damage. 
# Proper management includes monitoring blood sugar and maintaining a healthy lifestyle.
# In the modern world, artificial intelligence (AI) is changing everything around us. From voice assistants on our phones to self-driving cars, AI is becoming part of our daily lives.
# Terrorism continues to threaten peace and safety in many countries. Innocent people lose their lives, and communities are torn apart by violence and fear.
# Poverty is another problem that remains a huge burden across the world. In some countries, children don’t have enough food, clean water, or a safe place to sleep.
# """

# # semantic_chunker = SemanticChunker(initial.EMBEDDING_FUNCTION, breakpoint_threshold_type="percentile")
# # semantic_chunks = semantic_chunker.split_text(text)
# # for chunk in semantic_chunks:
# #     print(chunk)
# #     print("*"*100)

# # print("="*100)
# # print("*"*30, " Gradient ", "*"*30)
# # print("="*100)
# # semantic_chunker = SemanticChunker(initial.EMBEDDING_FUNCTION, breakpoint_threshold_type="gradient")
# # semantic_chunks = semantic_chunker.split_text(text)
# # for chunk in semantic_chunks:
# #     print(chunk)
# #     print("*"*100)


# # print("="*100)
# # print("*"*30, " InterQuartile ", "*"*30)
# # print("="*100)
# # semantic_chunker = SemanticChunker(initial.EMBEDDING_FUNCTION, breakpoint_threshold_type="interquartile")
# # semantic_chunks = semantic_chunker.split_text(text)
# # for chunk in semantic_chunks:
# #     print(chunk)
# #     print("*"*100)


# # print("="*100)
# # print("*"*30, " standard_deviation ", "*"*30)
# # print("="*100)
# # semantic_chunker = SemanticChunker(initial.EMBEDDING_FUNCTION, breakpoint_threshold_type="standard_deviation")
# # semantic_chunks = semantic_chunker.split_text(text)
# # for chunk in semantic_chunks:
# #     print(chunk)
# #     print("*"*100)


# # print("="*100)
# # print("*"*30, " Recurrsive text splitter ", "*"*30)
# # print("="*100)
# # splitter = RecursiveCharacterTextSplitter(
# #     chunk_size=500,
# #     chunk_overlap=100
# # )
# # splitts = splitter.split_text(text)
# # for chunk in splitts:
# #     print(chunk)
# #     print("*"*100)


from langchain_core.runnables import RunnableLambda

text = "This is a test string to stream."

streamable = RunnableLambda(lambda _: text)

for chunk in streamable.stream({}):
    print(chunk, end="", flush=True)
