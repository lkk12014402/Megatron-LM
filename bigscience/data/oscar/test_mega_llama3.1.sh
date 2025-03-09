# curl 'http://localhost:5000/api' -X 'PUT' -H 'Content-Type: application/json; charset=UTF-8'  -d '{"prompts":["what is the deep learning?"], "tokens_to_generate":100, "top_k":1}'
curl 'http://localhost:5000/api' -X 'PUT' -H 'Content-Type: application/json; charset=UTF-8'  -d '{"prompts":["what is the deep learning?"], "tokens_to_generate":32, "top_p": 0.9, "temperature":0.6}'
