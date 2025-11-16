import json

def load_queries(filepath):
    # on charge les paires requête-réponse depuis le fichier JSONL
    query_answer_pairs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                # on parse chaque ligne JSON
                data = json.loads(line)
                answer_file = data['Answer file']
                # on crée une paire pour chaque requête associée au même document
                for query in data['Queries']:
                    query_answer_pairs.append({
                        'query': query,
                        'answer': answer_file
                    })
    return query_answer_pairs