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
r = wcapi.get("orders", params={
    "per_page":100,
    "page":1,
    "_fields":"total,line_items",
    "status":"checkout-draft"
})

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

print("...........", r.json())

