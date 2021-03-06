import numpy as np

def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding="utf8") as f:
        words = set()
        word_to_vec_map = {}
        
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
            
    return words, word_to_vec_map

def cosine_similarity(u, v):
    distance = 0.0
    dot = np.dot(u,v)
    norm_u = np.sqrt(np.dot(u,u))
    norm_v = np.sqrt(np.dot(v,v))
    cosine_similarity = dot / (norm_u*norm_v)
    return cosine_similarity

def neutralize(word, g, word_to_vec_map):
    e = word_to_vec_map[word]
    e_biascomponent = (np.dot(e,g)/np.sqrt(np.dot(g,g))**2)*g
    e_debiased = e - e_biascomponent
    return e_debiased

def equalize(pair, bias_axis, word_to_vec_map):
    w1, w2 = pair[0], pair[1]
    e_w1, e_w2 = word_to_vec_map[w1], word_to_vec_map[w2]
    mu = (e_w1 + e_w2)/2
    mu_B = np.dot(mu, bias_axis)/(np.square(np.sqrt(np.dot(bias_axis, bias_axis)))) * bias_axis
    mu_orth = mu - mu_B
    e_w1B = np.dot(e_w1, bias_axis)/np.square(np.sqrt(np.dot(bias_axis, bias_axis))) * bias_axis
    e_w2B = np.dot(e_w2, bias_axis)/np.square(np.sqrt(np.dot(bias_axis, bias_axis))) * bias_axis
    corrected_e_w1B = np.sqrt(np.abs(1 - np.square(np.sqrt(np.dot(mu_orth, mu_orth))))) * (e_w1B - mu_B)/np.linalg.norm(((e_w1 - mu_orth) - mu_B))
    corrected_e_w2B = np.sqrt(np.abs(1 - np.square(np.sqrt(np.dot(mu_orth, mu_orth))))) * (e_w2B - mu_B)/np.linalg.norm(((e_w2 - mu_orth) - mu_B))
    e1 = corrected_e_w1B + mu_orth
    e2 = corrected_e_w2B + mu_orth
    return e1, e2
