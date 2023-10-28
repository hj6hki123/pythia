from transformers import GPTNeoXForCausalLM, AutoTokenizer
from tqdm import tqdm
import faiss
import glob
import numpy as np
from sentence_transformers import SentenceTransformer
from itertools import chain
import pickle


def load_sentence_vectors(sentences):
    sent_model = SentenceTransformer('all-mpnet-base-v2')
    return sent_model.encode(sentences)


def load_or_compute_vectors(vector_map_file, sentence_map_file, sentence_dir):
    try:
        vector_map = np.load(vector_map_file, allow_pickle=True).item()
        with open(sentence_map_file, 'rb') as f:
            sentence_map = pickle.load(f)
        print("Loaded pre-computed vectors from file.")
    except:
        print("file not found, start compute")
        vector_map = {}
        sentence_map = {}
        file_list = glob.glob(sentence_dir)

        # Create a progress bar for the loop
        with tqdm(total=len(file_list), desc="Computing Vectors") as pbar:
            for fname in file_list:
                with open(fname) as f:
                    sentence_map[fname] = f.readlines()
                    vector_map[fname] = load_sentence_vectors(sentence_map[fname])
                pbar.update(1)  # Update the progress bar

        # Save the results
        np.save(vector_map_file, vector_map)
        with open(sentence_map_file, 'wb') as f:
            pickle.dump(sentence_map, f)

        print("Saved sentence_map to file.")
        print("Saved computed vectors to file.")

    return vector_map, sentence_map


def build_and_index_faiss(vector_map):
    d = list(vector_map.values())[0].shape[1]
    index = faiss.IndexFlatL2(d)
    print("index is trained:", index.is_trained)
    for sent_vec in vector_map.values():
        index.add(sent_vec)
    print("index total:", index.ntotal)
    return index


def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
    input_ids = tokenizer.encode(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    generated_tokens_with_prompt = model.generate(
        input_ids=input_ids,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_output_tokens,
        
    )
    generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt)
    generated_text_answer = generated_text_with_prompt[0][len(text):]
    return generated_text_answer





if __name__ == '__main__':
    # vector_map, sentence_map = load_or_compute_vectors(r'temp/vector_map.npy', r'temp/sentence_map.pkl',
    #                                                    r"AICUP/First_Phase_Release(Correction)/First_Phase_Text_Dataset/*.txt")  # q
    # q_faiss_index = build_and_index_faiss(vector_map)

    # ans_vector_map, ans_sentence_map = load_or_compute_vectors(r'temp/ans_vector_map.npy', r'temp/ans_sentence_map.pkl',
    #                                                            r"AICUP/First_Phase_Release(Correction)/answer.txt")  # a
    # ans_faiss_index = build_and_index_faiss(ans_vector_map)

    # 載入模型
    model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-70m-deduped",
                                               cache_dir=r'./pythia-70m-deduped')

    # 載入 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped",
                                              cache_dir=r'./pythia-70m-deduped')

    # querys = ["JAXON AL-KARSTEN", "PICKETTE, TERRENCE"]
    # xq = load_sentence_vectors(querys)
    # D, I = q_faiss_index.search(xq, k=2)
    # AD , AI = ans_faiss_index.search(xq, k=2)
    # txts = list(chain.from_iterable(sentence_map.values()))
    # ans = list(chain.from_iterable(ans_sentence_map.values()))
    
    # neighbors = np.take(txts, I)
    # answers = np.take(ans, AI)

    # for qidx,q in enumerate(querys):
    #     print(qidx)
    #     for exs in neighbors:
    #         print(exs)

   

    print(inference("what are you",model,tokenizer))

    # for qidx , q in enumerate(querys):
    #     template = task_instruction_template
    #     for exs in neighbors:
    #         for idx, ex in enumerate(exs):
    #             example = example_template_replace('[__IN__]', ex)
    #             example = example.replace('[__OUT__]', answers[qidx,idx].replace("\\n","\t"))
    #             template = template + example
    #     template = template + query.replace('[__IN__]', q)
    #     print(template)
    #     print(inference(template,model,tokenizer))
