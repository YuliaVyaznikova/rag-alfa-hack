from generate import gen
from rag import Rag 
from parsedocs import from_pdf
from prompts import critic

text = from_pdf('./doc.pdf')

rag = Rag([text], chunk_size=200)

QUERY = 'what is convolutional neural network?'

rag_results = ''
for i, res in enumerate(rag.retrieve(QUERY)):
    rag_results += f'result {i}: ' + res + '\n\n'

print('RAG RESULTS')
print(rag_results)

critprompt = critic.format(results=rag_results, query=QUERY) 
print('CRITIC ANSWER:\n', gen(critprompt))