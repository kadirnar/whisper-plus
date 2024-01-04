from pipelines.pipeline.chatbot import ChatWithVideo


input_file = 'trascript.text'
llm_model_name = 'TheBloke/Mistral-7B-v0.1-GGUF'
llm_model_file = 'mistral-7b-v0.1.Q4_K_M.gguf'
llm_model_type = "mistral"
embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
chat = ChatWithVideo(input_file,llm_model_name, llm_model_file, llm_model_type, embedding_model_name)
query = "what is this video about ?"
response = chat.run_query(query)
print(response)
